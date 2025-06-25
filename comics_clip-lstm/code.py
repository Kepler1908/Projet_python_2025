import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from transformers import AutoProcessor, AutoModel
from datasets import load_dataset

# --- Chemins ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PRECOMPUTED_DIR = os.path.join(BASE_DIR, "precomputed_embeddings_clip-vit-base-patch32")
BEST_MODEL_PATH = os.path.join(BASE_DIR, "best_model.pt")
MODEL_NAME = "openai/clip-vit-base-patch32"

# --- Hyperparametres ---
IMAGE_BATCH_SIZE = 512
EMBEDDING_DIM = 768
LSTM_HIDDEN = 256
BATCH_SIZE = 128
LEARNING_RATE = 1e-4
NUM_EPOCHS = 50

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Embeddings ---
def process_and_save_batch(liste_images, liste_metadata, filepath, processor, model):
    if not liste_images:
        return
    with torch.no_grad():
        inputs = processor(images=liste_images, return_tensors="pt").to(device)
        all_embeds = model(pixel_values=inputs["pixel_values"]).pooler_output.cpu()

    start_idx = 0
    with open(filepath, 'a') as file:
        for metadata in liste_metadata:
            n_context = metadata['n_context']
            n_options = metadata['n_options']
            end_context = start_idx + n_context
            context_embeds = all_embeds[start_idx:end_context].tolist()
            start_options = end_context
            end_options = start_options + n_options
            option_embeds = all_embeds[start_options:end_options].tolist()
            data_to_save = {
                'context_embeds': context_embeds,
                'option_embeds': option_embeds,
                'label': metadata['label']
            }
            file.write(json.dumps(data_to_save) + '\n')
            start_idx = end_options

def create_embedding_dataset(split_name, raw_dataset, processor, vision_model):
    output_path = os.path.join(PRECOMPUTED_DIR, f"{split_name}.jsonl")
    os.makedirs(PRECOMPUTED_DIR, exist_ok=True)
    open(output_path, 'w').close() # Clear the file if it exists

    liste_images, liste_metadata = [], []
    for example in tqdm(raw_dataset, desc=f"Processing {split_name}"):
        label = example['solution_index']
        label = len(example['options']) + label if label < 0 else label
        context_imgs = example['context']
        option_imgs = example['options']
        liste_images.extend(context_imgs)
        liste_images.extend(option_imgs)
        liste_metadata.append({
            'n_context': len(context_imgs),
            'n_options': len(option_imgs),
            'label': label
        })
        if len(liste_images) >= IMAGE_BATCH_SIZE:
            process_and_save_batch(liste_images, liste_metadata, output_path, processor, vision_model)
            liste_images, liste_metadata = [], []

    process_and_save_batch(liste_images, liste_metadata, output_path, processor, vision_model)

# --- Dataset Loader ---
class PrecomputedEmbeddingDataset(Dataset):
    def __init__(self, jsonl_path):
        self.data = []
        with open(jsonl_path, 'r') as f:
            for line in f:
                self.data.append(json.loads(line))

    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        item = self.data[idx]
        context = torch.tensor(item['context_embeds'], dtype=torch.float)
        options = torch.tensor(item['option_embeds'], dtype=torch.float)
        label = torch.tensor(item['label'], dtype=torch.long)
        return {'context': context, 'options': options, 'label': label}

def collate_fn_fast(batch):
    contexts = [item['context'] for item in batch]
    options = [item['options'] for item in batch]
    labels = torch.stack([item['label'] for item in batch])
    context_batch = pad_sequence(contexts, batch_first=True, padding_value=0.0)
    option_batch = pad_sequence(options, batch_first=True, padding_value=0.0)
    return {'context': context_batch, 'options': option_batch, 'label': labels}

# --- Modèle ---
class ComicSequenceModel(nn.Module):
    def __init__(self, embedding_dim, lstm_hidden, dropout_rate=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=lstm_hidden, batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.option_proj = nn.Linear(embedding_dim, lstm_hidden)

    def forward(self, context_embeds, option_embeds):
        _, (h_n, _) = self.lstm(context_embeds)
        context_vector = self.dropout(h_n.squeeze(0))
        option_proj = self.dropout(self.option_proj(option_embeds))
        context_vector = F.normalize(context_vector, dim=-1).unsqueeze(1)
        option_proj = F.normalize(option_proj, dim=-1)
        context_expanded = context_vector.expand(-1, option_proj.size(1), -1)
        return F.cosine_similarity(option_proj, context_expanded, dim=-1)

# --- Entraînement et validation ---
def train_model():
    train_path = os.path.join(PRECOMPUTED_DIR, "train.jsonl")
    val_path = os.path.join(PRECOMPUTED_DIR, "val.jsonl")

    if not os.path.exists(train_path) or not os.path.exists(val_path):
        print(f"Erreur: les fichiers d'embeddings pré-calculés sont introuvables dans {PRECOMPUTED_DIR}")
        print("Indiquer --preprocess pour lancer le pré-traitement")
        return

    train_dataset = PrecomputedEmbeddingDataset(train_path)
    val_dataset = PrecomputedEmbeddingDataset(val_path)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn_fast, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn_fast, num_workers=2, pin_memory=True)

    model = ComicSequenceModel(EMBEDDING_DIM, LSTM_HIDDEN).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda'))
    best_val_acc = 0.0

    for epoch in range(NUM_EPOCHS):
        # --- Entrainement ---
        model.train()
        total_loss, correct_train, total_train = 0, 0, 0
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} - Train")
        for batch in train_pbar:
            context = batch['context'].to(device, non_blocking=True)
            options = batch['options'].to(device, non_blocking=True)
            labels = batch['label'].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type='cuda', enabled=(device.type == 'cuda')):
                outputs = model(context, options)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            correct_train += (preds == labels).sum().item()
            total_train += labels.size(0)
            train_pbar.set_postfix(loss=f"{loss.item():.4f}")

        train_acc = correct_train / total_train
        train_loss = total_loss / len(train_loader)

        # --- Validation ---
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} - Val"):
                context = batch['context'].to(device, non_blocking=True)
                options = batch['options'].to(device, non_blocking=True)
                labels = batch['label'].to(device, non_blocking=True)
                with torch.amp.autocast(device_type='cuda', enabled=(device.type == 'cuda')):
                    outputs = model(context, options)
                    preds = torch.argmax(outputs, dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_acc = val_correct / val_total
        
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} Summary:")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"  Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f"  -> Chemin du nouveau meilleur modèle {BEST_MODEL_PATH} (Val Acc: {best_val_acc:.4f})")

    print("Training complete.")

def preprocess_data():
    print("--- Starting Data Preprocessing ---")
    print(f"Loading vision model: {MODEL_NAME}")
    vision_model = AutoModel.from_pretrained(MODEL_NAME).vision_model.to(device)
    processor = AutoProcessor.from_pretrained(MODEL_NAME, use_fast=True)
    vision_model.eval()

    print("Processing 'train' split...")
    train_raw_dataset = load_dataset("VLR-CVC/ComicsPAP", "sequence_filling", split="train", streaming=True)
    create_embedding_dataset("train", train_raw_dataset, processor, vision_model)

    print("Processing 'val' split...")
    val_raw_dataset = load_dataset("VLR-CVC/ComicsPAP", "sequence_filling", split="val", streaming=True)
    create_embedding_dataset("val", val_raw_dataset, processor, vision_model)

    print(f"Pré-traitement terminé!. Embeddings enregistré dans '{PRECOMPUTED_DIR}'")

# --- Main ---
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train a Comic Sequence Model.")
    parser.add_argument("--preprocess", action="store_true")
    parser.add_argument("--train", action="store_true")
    args = parser.parse_args()

    if args.preprocess:
        preprocess_data()
    
    if args.train:
        train_model()

    if not args.preprocess and not args.train:
        print("Indiquer --preprocess et/ou --train.")
        print("Exemple: python your_script_name.py --preprocess --train")
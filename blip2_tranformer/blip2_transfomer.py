import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import AutoModelForVisualQuestionAnswering, AutoProcessor, AutoTokenizer, AutoModel
from PIL import Image
import io
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from datasets import load_dataset
import torch.multiprocessing as mp
import gc
import pandas as pd
from spellchecker import SpellChecker
from textblob import TextBlob
import language_tool_python
import re
from torchvision import transforms
import random

# Set environment variables
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# Global model references (to be initialized in initialize_models)
blip_processor = None
blip_model = None
text_tokenizer = None
text_encoder = None

train_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    transforms.RandomAffine(degrees=5, translate=(0.05, 0.05), scale=(0.95, 1.05)),
])


def clear_gpu_memory():
    """Clear GPU memory cache and run garbage collection"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()

def initialize_models(device):
    """Initialize all models needed for feature extraction"""
    global blip_processor, blip_model, text_tokenizer, text_encoder
    
    print("Loading pre-trained models...")
    
    # BLIP-2 for image embedding
    try:
        blip_processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
        blip_model = AutoModelForVisualQuestionAnswering.from_pretrained("Salesforce/blip2-opt-2.7b")
        blip_model = blip_model.to(device).eval()
    except RuntimeError:
        print("Failed to load BLIP model to GPU, trying with reduced precision...")
        clear_gpu_memory()
        try:
            blip_model = AutoModelForVisualQuestionAnswering.from_pretrained(
                "Salesforce/blip2-opt-2.7b",
                torch_dtype=torch.float16
            ).to(device).eval()
        except RuntimeError:
            print("GPU loading still failed, falling back to CPU")
            device = torch.device("cpu")
            blip_model = AutoModelForVisualQuestionAnswering.from_pretrained("Salesforce/blip2-opt-2.7b").to(device).eval()

    # Text encoder (MiniLM)
    text_tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L12-v2")
    text_encoder = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L12-v2").to(device).eval()

    # Ensure models are in eval mode
    print("Models loaded successfully!")
    return device

def get_image_embedding(img, device):
    """Extract image embeddings using BLIP model"""
    global blip_processor, blip_model
    
    # Resize image if needed
    if max(img.size) > 800:
        # Keep aspect ratio
        ratio = min(800 / img.size[0], 800 / img.size[1])
        new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
        img = img.resize(new_size, Image.LANCZOS)
    
    inputs = blip_processor(images=img, return_tensors="pt").to(device)
    with torch.no_grad():
        vision_out = blip_model.vision_model(**inputs)
        emb = vision_out.last_hidden_state.mean(dim=1).squeeze().cpu()  # (embed_dim,)
    return emb

def get_text_embedding(text, device):
    """Extract text embeddings using MiniLM"""
    global text_tokenizer, text_encoder
    
    if not isinstance(text, str):
        text = ""  # Empty string for non-string inputs
    
    inputs = text_tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
    with torch.no_grad():
        output = text_encoder(**inputs)
        emb = output.last_hidden_state.mean(dim=1).squeeze().cpu()  # (text_dim,)
    return emb

def fuse_modalities(img_emb, text_emb):
    """Combine image and text embeddings"""
    combined = torch.cat([img_emb, text_emb], dim=-1)
    return combined

# Transformer-based classifier
class MultimodalTransformerClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=512, nhead=8, num_layers=4, num_choices=5, max_context_len=5):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Ensure hidden_dim is divisible by nhead
        if hidden_dim % nhead != 0:
            hidden_dim = (hidden_dim // nhead) * nhead
            print(f"Adjusted hidden_dim to {hidden_dim} to make it divisible by nhead={nhead}")

        self.positional_embeddings = nn.Embedding(max_context_len, hidden_dim)
        
        # Project input to hidden dimension first
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Projections and classifier
        decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=nhead, batch_first=True)
        self.cross_attention_decoder = nn.TransformerDecoder(decoder_layer, num_layers=1) # Use one layer for a start
        
        self.option_proj = nn.Linear(input_dim, hidden_dim) # We still need to project options
        self.final_classifier = nn.Linear(hidden_dim, 1)


        self.dropout = nn.Dropout(0.1)
        self.num_choices = num_choices

    def forward(self, context_embeds, options_embeds):
        # context_embeds: (B, seq_len, input_dim)
        # batch_size = context_embeds.size(0)

        batch_size, seq_len, _ = context_embeds.size()
        device = context_embeds.device
        
        # Project to hidden dimension
        context_projected = self.input_projection(context_embeds)  # (B, seq_len, hidden_dim)
        position_ids = torch.arange(seq_len, dtype=torch.long, device=device).unsqueeze(0).expand(batch_size, -1)
        position_embeds = self.positional_embeddings(position_ids)
        context_with_pos = context_projected + position_embeds


        # Apply transformer
        context_out = self.transformer(context_with_pos)  # (B, seq_len, hidden_dim)
        
        # Get context summary - mean across sequence dimension
        # context_summary = context_out.mean(dim=1)  # (B, hidden_dim)
        # context_summary = self.dropout(context_summary)
        # context_summary = self.cls_proj(context_summary)  # (B, hidden_dim)
        
        logits = []
        for i in range(min(self.num_choices, options_embeds.size(1))):
            # options_embeds: (B, num_choices, input_dim)
            option = options_embeds[:, i, :]  # (B, input_dim)
            opt_vec = self.option_proj(option).unsqueeze(1)  # (B, hidden_dim)
            
            # Element-wise multiplication then classifier
            interaction_output = self.cross_attention_decoder(tgt=opt_vec, memory=context_out)
            interaction_output = self.dropout(interaction_output.squeeze(1)) # Back to (B, hidden_dim)
            score = self.final_classifier(interaction_output) # (B, 1)
            logits.append(score)


        return torch.cat(logits, dim=1)  # (B, num_choices)

def precompute_embeddings(samples, device, max_context_len=5, cache_dir=None, 
                         ocr_correction_method=None, augment=True):
    """
    Enhanced version with OCR correction integrated.
    
    Args:
        samples: Your comic dataset samples
        device: PyTorch device
        max_context_len: Maximum context length
        cache_dir: Cache directory for processed samples
        ocr_correction_method: Method for OCR correction
            - "spell_checker": Fast, basic correction
            - "textblob": Balanced speed and accuracy
            - "language_tool": Slow but most accurate
            - "hybrid": Best results but slowest
            - "none": No correction (original behavior)
    """
    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)
    

    
    processed_samples = []
    
    for idx, item in enumerate(tqdm(samples, desc="Precomputing embeddings")):
        if cache_dir is not None:
            cache_path = os.path.join(cache_dir, f"item_{idx}.pt")
            if os.path.exists(cache_path):
                processed_samples.append({"idx": idx, "cache_path": cache_path})
                continue
        
        try:
            context_images = item["context"]
            missing_index = item["index"]
            
            # Process context panels (excluding the missing panel)
            context_panels = []
            for i in range(len(context_images)):
                if i != missing_index:
                    try:
                        img = context_images[i]

                        if augment:
                            # Apply data augmentation
                            img = train_transforms(img)
                        
                        # Get text for this panel - defaults to empty string
                        ocr_text = ""
                        if hasattr(item, "context_text_vlm") and item["context_text_vlm"] and \
                           isinstance(item["context_text_vlm"], list) and i < len(item["context_text_vlm"]) and \
                           isinstance(item["context_text_vlm"][i], dict) and "panel_text_ordered" in item["context_text_vlm"][i]:
                            ocr_text = item["context_text_vlm"][i]["panel_text_ordered"]
                        
                        
                        img_emb = get_image_embedding(img, device)
                        txt_emb = get_text_embedding(ocr_text, device)
                        vec = fuse_modalities(img_emb, txt_emb)
                        context_panels.append(vec)
                    except Exception as e:
                        print(f"Error processing context panel {i} in sample {idx}: {e}")
                        # Use zero vector as fallback
                        placeholder_dim = get_image_embedding(Image.new("RGB", (224, 224)), device).shape[0] + \
                                          get_text_embedding("placeholder", device).shape[0]
                        context_panels.append(torch.zeros(placeholder_dim))
            
            # Handle padding for consistent context length
            if len(context_panels) < max_context_len:
                pad_len = max_context_len - len(context_panels)
                if len(context_panels) > 0:
                    pad_dim = context_panels[0].shape[0]
                    pad_tensor = torch.zeros((pad_len, pad_dim))
                    context_tensor = torch.cat([torch.stack(context_panels), pad_tensor], dim=0)
                else:
                    # Handle edge case where no valid context panels
                    placeholder_dim = get_image_embedding(Image.new("RGB", (224, 224)), device).shape[0] + \
                                      get_text_embedding("placeholder", device).shape[0]
                    context_tensor = torch.zeros((max_context_len, placeholder_dim))
            else:
                context_tensor = torch.stack(context_panels[:max_context_len])

            # Process option panels
            option_vecs = []
            option_images = item["options"]
            for i in range(len(option_images)):
                try:
                    img = option_images[i]

                    if augment:
                        # Apply data augmentation
                        img = train_transforms(img)
                    
                    # Get text for this option - defaults to empty string
                    ocr_text = ""
                    if hasattr(item, "options_text_vlm") and item["options_text_vlm"] and \
                       isinstance(item["options_text_vlm"], list) and i < len(item["options_text_vlm"]) and \
                       isinstance(item["options_text_vlm"][i], dict) and "panel_text_ordered" in item["options_text_vlm"][i]:
                        ocr_text = item["options_text_vlm"][i]["panel_text_ordered"]
                    
                    
                    img_emb = get_image_embedding(img, device)
                    txt_emb = get_text_embedding(ocr_text, device)
                    vec = fuse_modalities(img_emb, txt_emb)
                    option_vecs.append(vec)
                except Exception as e:
                    print(f"Error processing option {i} in sample {idx}: {e}")
                    placeholder_dim = get_image_embedding(Image.new("RGB", (224, 224)), device).shape[0] + \
                                     get_text_embedding("placeholder", device).shape[0]
                    option_vecs.append(torch.zeros(placeholder_dim))
            
            options_tensor = torch.stack(option_vecs)  # (num_choices, input_dim)
            
            # Prepare the result
            result = {
                "context": context_tensor,
                "options": options_tensor,
                "label": torch.tensor(item["solution_index"], dtype=torch.long),
                "idx": idx # Store original index for potential cache loading identification
            }
            
            # Cache the result if cache_dir is specified
            if cache_dir is not None:
                cache_path = os.path.join(cache_dir, f"item_{idx}.pt")
                torch.save(result, cache_path)
                # For newly processed and cached items, still return the full result
                # but mark it as cached for potential future reference
                result["cache_path"] = cache_path
            
            processed_samples.append(result)
            
            # Clear memory
            if idx % 100 == 0:
                clear_gpu_memory()
                
        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
            continue
    
    return processed_samples

# Simplified dataset class for cached embeddings
class CachedEmbeddingsDataset(Dataset):
    def __init__(self, processed_samples_or_indices, cache_dir=None):
        self.samples = processed_samples_or_indices
        self.cache_dir = cache_dir
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        item_info = self.samples[idx] # This could be a dict with just "idx" or the full processed sample
        
        # Check if this is a cache reference (has "idx" and "cache_path" but not "context")
        if isinstance(item_info, dict) and "cache_path" in item_info and "context" not in item_info:
            # Load from cache using the cache_path
            cache_path = item_info["cache_path"]
            if os.path.exists(cache_path):
                return torch.load(cache_path)
            else:
                raise FileNotFoundError(f"Cache file not found: {cache_path}")
        
        # Check if this is an old-style cache reference (has "idx" but not "context" and no "cache_path")
        elif self.cache_dir is not None and isinstance(item_info, dict) and "idx" in item_info and "context" not in item_info:
            # Load from cache using the original index stored in item_info["idx"]
            cache_path = os.path.join(self.cache_dir, f"item_{item_info['idx']}.pt")
            if os.path.exists(cache_path):
                return torch.load(cache_path)
            else:
                raise FileNotFoundError(f"Cache file not found: {cache_path}")
        else:
            # Return directly if already processed and passed in full
            return item_info

# Collate function for batching
def collate_fn(batch):

    # Find the maximum number of options in the batch
    max_options = 0
    for item in batch:
        if item['options'].size(0) > max_options:
            max_options = item['options'].size(0)
    if max_options == 0 and len(batch) > 0: # Fallback if all options are empty for some reason
        # This case should ideally not happen with valid data.
        # If it does, we need a default dimension for options.
        # For now, let's try to infer from first item or set a default.
        if batch[0]['options'].ndim > 1:
             placeholder_option_dim = batch[0]['options'].size(1)
        else: # Should not happen if options are embeddings
             placeholder_option_dim = 1 
        print(f"Warning: max_options is 0. Using placeholder dimension: {placeholder_option_dim}")


    # Initialize lists for contexts, padded options, and labels
    contexts = []
    options_list = []
    labels = []
    
    for item in batch:
        contexts.append(item['context'])
        options = item['options']
        num_options = options.size(0)
        
        # If the number of options is less than max_options, pad with zeros
        if num_options < max_options:
            # Ensure options has at least 2 dimensions for size(1)
            if options.ndim < 2: # Should be (num_options, feature_dim)
                # This is an error state, means options are not proper embeddings
                # Create a zero tensor of appropriate shape if possible
                print(f"Warning: Item options has unexpected ndim {options.ndim}. Shape: {options.shape}")
                # Attempt to get feature_dim if first dimension is 0
                feature_dim = options.size(0) if options.ndim == 1 and num_options == 0 and max_options > 0 else 1
                if item['options'].ndim > 1 : feature_dim = item['options'].size(1) # try to get from original item
                padding = torch.zeros(max_options - num_options, feature_dim)


            else:
                padding = torch.zeros(max_options - num_options, options.size(1))

            if num_options == 0: # If there were no options to begin with
                 padded_options = padding
            else:
                 padded_options = torch.cat([options, padding], dim=0)
        else:
            padded_options = options
        
        options_list.append(padded_options)
        labels.append(item['label'])

    contexts = torch.stack(contexts)
    options = torch.stack(options_list)
    labels = torch.stack(labels)
    
    return {
        'context': contexts,
        'options': options,
        'label': labels
    }

# Training function
def train(model, dataloader, optimizer, loss_fn, device, epoch, log_interval=10, accumulation_steps=1):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    start_time = time.time()

    optimizer.zero_grad()
    
    for batch_idx, batch in enumerate(dataloader):
        try:
            # Move batch to device
            context = batch["context"].to(device)        # (B, seq_len, input_dim)
            options = batch["options"].to(device)        # (B, num_choices, input_dim)
            labels = batch["label"].to(device)           # (B,)
            
            # Forward pass
            optimizer.zero_grad()
            logits = model(context, options)             # (B, num_choices)
            loss = loss_fn(logits, labels)
            loss = loss / accumulation_steps
            
            # Backward pass
            loss.backward()

            if (batch_idx + 1) % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
            
            # Clip gradients to prevent exploding gradients
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # optimizer.step()

            # Update metrics
            total_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            batch_correct = (predicted == labels).sum().item()
            correct += batch_correct
            total += labels.size(0)
            
            # Log progress
            if batch_idx % log_interval == 0:
                elapsed = time.time() - start_time
                print(f"Epoch {epoch} | Batch {batch_idx}/{len(dataloader)} | "
                      f"Loss: {loss.item():.4f} | Acc: {100 * batch_correct / labels.size(0):.2f}% | "
                      f"Time: {elapsed:.2f}s")
                start_time = time.time()
                
        except Exception as e:
            print(f"Error in batch {batch_idx}: {e}")
            # Optionally clear GPU memory if an error like OOM occurs
            clear_gpu_memory()
            continue
            
    accuracy = 100 * correct / total if total > 0 else 0
    avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else float('inf')
    return avg_loss, accuracy

# Validation function
def validate(model, dataloader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in dataloader:
            try:
                context = batch["context"].to(device)
                options = batch["options"].to(device)
                labels = batch["label"].to(device)
                
                logits = model(context, options)
                loss = loss_fn(logits, labels)
                
                total_loss += loss.item()
                _, predicted = torch.max(logits, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
            except Exception as e:
                print(f"Error during validation: {e}")
                clear_gpu_memory()
                continue
            
    accuracy = 100 * correct / total if total > 0 else 0
    avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else float('inf')
    return avg_loss, accuracy

# Plot training history
def plot_training_history(train_losses, val_losses, train_accs, val_accs, save_path=None):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot losses
    ax1.plot(train_losses, label='Train Loss')
    ax1.plot(val_losses, label='Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    
    # Plot accuracies
    ax2.plot(train_accs, label='Train Accuracy')
    ax2.plot(val_accs, label='Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close(fig) # Close the figure to free memory

def prepare_comic_dataset(dataset_stream, current_split_name, max_samples=None, ocr_df=None):
    """
    Convert streaming dataset to our format and prepare it for training.
    Filters OCR data based on current_split_name.
    """
    samples = []
    
    # Convert streaming dataset to list (with limit if specified)
    for idx, item in enumerate(tqdm(dataset_stream, desc=f"Processing dataset for {current_split_name} split")):
        if max_samples is not None and idx >= max_samples:
            break
            
        # If we have OCR data available, add it to the sample if the split matches
        if ocr_df is not None:
            sample_id_str = str(item.get("sample_id", "")) # Use .get for safety
            if sample_id_str and sample_id_str in ocr_df.index: # Check if sample_id_str is not empty and exists in index
                ocr_data_for_id = ocr_df.loc[sample_id_str] # This can be Series or DataFrame

                selected_ocr_entry_data = None # To store context_text, options_text from the chosen entry

                if isinstance(ocr_data_for_id, pd.DataFrame): # Multiple entries for this sample_id
                    if not ocr_data_for_id.empty:
                        # Iterate through these multiple entries to find one matching current_split_name
                        for _index, row_series in ocr_data_for_id.iterrows(): # Use _index to avoid confusion with outer idx
                            if "split" in row_series and isinstance(row_series["split"], str) and row_series["split"] == current_split_name:
                                # Found a matching row
                                selected_ocr_entry_data = {
                                    "context_text_vlm": row_series.get("context_text_vlm"),
                                    "options_text_vlm": row_series.get("options_text_vlm")
                                }
                                break # Take the first one that matches
                elif isinstance(ocr_data_for_id, pd.Series): # Single entry for this sample_id
                    row_series = ocr_data_for_id
                    if "split" in row_series and isinstance(row_series["split"], str) and row_series["split"] == current_split_name:
                        # The single entry matches the current_split_name
                        selected_ocr_entry_data = {
                            "context_text_vlm": row_series.get("context_text_vlm"),
                            "options_text_vlm": row_series.get("options_text_vlm")
                        }
                
                if selected_ocr_entry_data is not None:
                    # item["context_text"] / item["options_text"] will be populated if get() found the keys
                    # and their values were not None. If keys are missing, .get() returns None.
                    if selected_ocr_entry_data.get("context_text_vlm") is not None:
                         item["context_text_vlm"] = selected_ocr_entry_data["context_text_vlm"]
                    if selected_ocr_entry_data.get("options_text_vlm") is not None:
                         item["options_text_vlm"] = selected_ocr_entry_data["options_text_vlm"]
        
        samples.append(item)
    
    return samples


class EarlyStopping:
    def __init__(self, patience=7, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_score):
        if self.best_score is None:
            self.best_score = val_score
        elif val_score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_score
            self.counter = 0
        return self.early_stop

def train_with_full_dataset_chunks(dataset_name="VLR-CVC/ComicsPAP", task="sequence_filling", 
                          batch_size=20, learning_rate=5e-5, hidden_dim=1024, 
                          chunk_size=1000, num_epochs_per_chunk=3,  
                          num_passes=2, save_dir="model_checkpoints", 
                          ocr_df=None, replay_buffer_size=500): 


    os.makedirs(save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    device = initialize_models(device)
    
    # Cache directories
    base_cache_dir = os.path.join(save_dir, "embeddings_cache_full_chunks")
    val_cache_dir = os.path.join(base_cache_dir, "val")
    train_chunks_base_cache_dir = os.path.join(base_cache_dir, "train_chunks")
    os.makedirs(val_cache_dir, exist_ok=True)
    os.makedirs(train_chunks_base_cache_dir, exist_ok=True)

    dummy_img = Image.new("RGB", (224, 224))
    img_dim = get_image_embedding(dummy_img, device).shape[0]
    text_dim = get_text_embedding("dummy text", device).shape[0]
    del dummy_img
    input_dim = img_dim + text_dim
    print(f"Input dimension: {input_dim} (Image: {img_dim}, Text: {text_dim})")
    
    model = MultimodalTransformerClassifier(
        input_dim=input_dim, hidden_dim=hidden_dim, num_choices=5
    ).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)


    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.7, patience=5
    )
    
    loss_fn = nn.CrossEntropyLoss()
    
    all_train_losses, all_val_losses, all_train_accs, all_val_accs = [], [], [], []
    best_val_acc = 0.0
    

    replay_buffer = []
    
    # Validation dataset
    print("Loading and precomputing full validation dataset...")
    val_stream = load_dataset(dataset_name, task, split="val", streaming=True)
    val_samples = prepare_comic_dataset(val_stream, "val", max_samples=None, ocr_df=ocr_df)
    del val_stream; clear_gpu_memory()
    
    val_processed = precompute_embeddings(val_samples, device, cache_dir=val_cache_dir, augment=False)
    del val_samples; clear_gpu_memory()
    val_dataset = CachedEmbeddingsDataset(val_processed, cache_dir=val_cache_dir)
    del val_processed; clear_gpu_memory()
    val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=collate_fn, 
                          num_workers=0, pin_memory=True)
    

    print("Collecting validation samples for regularization...")
    val_samples_for_reg = []
    for i, batch in enumerate(val_loader):
        if i >= 5:  # 只取前5个batch
            break
        val_samples_for_reg.extend([{
            'context': batch['context'][j], 
            'options': batch['options'][j], 
            'label': batch['label'][j]
        } for j in range(batch['context'].size(0))])
    
    global_epoch_counter = 0
    
    for pass_idx in range(num_passes):
        print(f"\n=== Starting Pass {pass_idx+1}/{num_passes} through the training dataset ===")
        train_stream = load_dataset(dataset_name, task, split="train", streaming=True)
        train_iterator = iter(train_stream)
        chunk_idx_this_pass = 0
        samples_processed_in_pass = 0
        
        while True:
            print(f"\n--- Processing Training Chunk {chunk_idx_this_pass+1} (Pass {pass_idx+1}) ---")
            chunk_samples = []
            for _ in range(chunk_size):
                try:
                    item = next(train_iterator)
                    samples_processed_in_pass += 1
  
                    if ocr_df is not None:
                        sample_id_str = str(item.get("sample_id", ""))
                        if sample_id_str and sample_id_str in ocr_df.index:
                            ocr_data_for_id = ocr_df.loc[sample_id_str]
                            
                            selected_ocr_entry_data = None

                            if isinstance(ocr_data_for_id, pd.DataFrame):
                                if not ocr_data_for_id.empty:
                                    for _index, row_series in ocr_data_for_id.iterrows():
                                        if "split" in row_series and isinstance(row_series["split"], str) and row_series["split"] == "train": 
                                            selected_ocr_entry_data = {
                                                "context_text_vlm": row_series.get("context_text_vlm"),
                                                "options_text_vlm": row_series.get("options_text_vlm")
                                            }
                                            break 
                            elif isinstance(ocr_data_for_id, pd.Series):
                                row_series = ocr_data_for_id
                                if "split" in row_series and isinstance(row_series["split"], str) and row_series["split"] == "train":
                                    selected_ocr_entry_data = {
                                        "context_text_vlm": row_series.get("context_text_vlm"),
                                        "options_text_vlm": row_series.get("options_text_vlm")
                                    }

                            if selected_ocr_entry_data is not None:
                                if selected_ocr_entry_data.get("context_text_vlm") is not None:
                                    item["context_text_vlm"] = selected_ocr_entry_data["context_text_vlm"]
                                if selected_ocr_entry_data.get("options_text_vlm") is not None:
                                    item["options_text_vlm"] = selected_ocr_entry_data["options_text_vlm"]
                    
                    chunk_samples.append(item)
                except StopIteration:
                    break
            
            if not chunk_samples:
                break
            

            current_chunk_cache_dir = os.path.join(train_chunks_base_cache_dir, f"pass_{pass_idx}_chunk_{chunk_idx_this_pass}")
            os.makedirs(current_chunk_cache_dir, exist_ok=True)
            
            processed_samples_in_chunk = precompute_embeddings(chunk_samples, device, cache_dir=current_chunk_cache_dir)
            del chunk_samples; clear_gpu_memory()

            mixed_samples = processed_samples_in_chunk.copy()
            if replay_buffer:
                replay_samples = random.sample(replay_buffer, min(len(replay_buffer), len(processed_samples_in_chunk) // 3))
                mixed_samples.extend(replay_samples)
                print(f"Added {len(replay_samples)} samples from replay buffer")
            

            if len(replay_buffer) < replay_buffer_size:
                replay_buffer.extend(processed_samples_in_chunk[:replay_buffer_size - len(replay_buffer)])
            else:

                replace_count = min(len(processed_samples_in_chunk), replay_buffer_size // 4)
                replace_indices = random.sample(range(len(replay_buffer)), replace_count)
                for i, idx in enumerate(replace_indices):
                    if i < len(processed_samples_in_chunk):
                        replay_buffer[idx] = processed_samples_in_chunk[i]
            
            chunk_dataset = CachedEmbeddingsDataset(mixed_samples, cache_dir=current_chunk_cache_dir)
            del processed_samples_in_chunk, mixed_samples; clear_gpu_memory()

            if not chunk_dataset or len(chunk_dataset) == 0:
                chunk_idx_this_pass += 1
                continue

            if pass_idx == 0 and chunk_idx_this_pass == 0:
                first_sample_data = chunk_dataset[0]
                if first_sample_data and "options" in first_sample_data and first_sample_data["options"] is not None:
                    num_options = first_sample_data["options"].size(0)
                    if num_options != model.num_choices:
                        print(f"Updating model with correct number of options: {num_options}")
                        model = MultimodalTransformerClassifier(
                            input_dim=input_dim, hidden_dim=hidden_dim, num_choices=num_options
                        ).to(device)
                        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
                        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                            optimizer, mode='max', factor=0.7, patience=5
                        )
            
            train_loader = DataLoader(
                chunk_dataset, batch_size=batch_size, shuffle=True, 
                collate_fn=collate_fn, num_workers=0, pin_memory=True
            )
            
            for epoch_in_chunk in range(num_epochs_per_chunk):
                global_epoch_counter += 1
                
                train_loss, train_acc = train_with_regularization(
                    model, train_loader, val_samples_for_reg, optimizer, loss_fn, device, 
                    global_epoch_counter, reg_weight=0.1
                )
                clear_gpu_memory()

                if epoch_in_chunk % 2 == 1 or epoch_in_chunk == num_epochs_per_chunk - 1:
                    val_loss, val_acc = validate(model, val_loader, loss_fn, device)
                    scheduler.step(val_acc)
                    
                    all_train_losses.append(train_loss); all_train_accs.append(train_acc)
                    all_val_losses.append(val_loss); all_val_accs.append(val_acc)
                    
                    print(f"Pass {pass_idx+1}, Chunk {chunk_idx_this_pass+1}, Epoch {epoch_in_chunk+1}")
                    print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
                    print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
                    print(f"  Learning rate: {optimizer.param_groups[0]['lr']:.6f}")
                    
                    if val_acc > best_val_acc:
                        best_val_acc = val_acc
                        model_path = os.path.join(save_dir, f"best_model_acc{val_acc:.2f}.pt")
                        torch.save({
                            'pass': pass_idx + 1, 'chunk': chunk_idx_this_pass + 1, 
                            'epoch_in_chunk': epoch_in_chunk + 1, 'global_epoch': global_epoch_counter,
                            'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(),
                            'val_acc': val_acc, 'input_dim': input_dim, 'hidden_dim': hidden_dim, 
                            'num_options': model.num_choices,
                        }, model_path)
                        print(f"  New best model saved")
            
            del chunk_dataset, train_loader
            clear_gpu_memory()
            chunk_idx_this_pass += 1

        del train_stream, train_iterator
        clear_gpu_memory()

    plot_training_history(all_train_losses, all_val_losses, all_train_accs, all_val_accs, 
                         save_path=os.path.join(save_dir, "training_history_improved.png"))
    
    return model, (all_train_losses, all_val_losses, all_train_accs, all_val_accs)


def train_with_regularization(model, train_loader, val_samples_for_reg, optimizer, loss_fn, device, 
                            epoch, reg_weight=0.1, log_interval=10, accumulation_steps=1):

    import random
    
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    start_time = time.time()

    optimizer.zero_grad()
    
    for batch_idx, batch in enumerate(train_loader):
        try:
            context = batch["context"].to(device)
            options = batch["options"].to(device)
            labels = batch["label"].to(device)
            

            logits = model(context, options)
            main_loss = loss_fn(logits, labels)
            

            reg_loss = 0.0
            if val_samples_for_reg and len(val_samples_for_reg) > 0:
                reg_batch_size = min(4, len(val_samples_for_reg))
                reg_samples = random.sample(val_samples_for_reg, reg_batch_size)
                
                reg_context = torch.stack([s['context'] for s in reg_samples]).to(device)
                reg_options = torch.stack([s['options'] for s in reg_samples]).to(device)
                reg_labels = torch.stack([s['label'] for s in reg_samples]).to(device)
                
                reg_logits = model(reg_context, reg_options)
                reg_loss = loss_fn(reg_logits, reg_labels)
            

            total_batch_loss = main_loss + reg_weight * reg_loss
            total_batch_loss = total_batch_loss / accumulation_steps
            
            total_batch_loss.backward()

            if (batch_idx + 1) % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
            

            total_loss += main_loss.item()
            _, predicted = torch.max(logits, 1)
            batch_correct = (predicted == labels).sum().item()
            correct += batch_correct
            total += labels.size(0)
            
            if batch_idx % log_interval == 0:
                elapsed = time.time() - start_time
                print(f"Epoch {epoch} | Batch {batch_idx}/{len(train_loader)} | "
                      f"Main Loss: {main_loss.item():.4f} | Reg Loss: {reg_loss.item() if isinstance(reg_loss, torch.Tensor) else reg_loss:.4f} | "
                      f"Acc: {100 * batch_correct / labels.size(0):.2f}% | Time: {elapsed:.2f}s")
                start_time = time.time()
                
        except Exception as e:
            print(f"Error in batch {batch_idx}: {e}")
            clear_gpu_memory()
            continue
            
    accuracy = 100 * correct / total if total > 0 else 0
    avg_loss = total_loss / len(train_loader) if len(train_loader) > 0 else float('inf')
    return avg_loss, accuracy




def predict_on_test_set(model, dataset_name="VLR-CVC/ComicsPAP", task="sequence_filling", 
                       batch_size=16, save_dir="model_checkpoints", ocr_df=None):
    """
    Run predictions on the test set and evaluate performance
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    # No need to initialize models if they are global and already initialized for the model being passed
    # If this function can be called standalone, then initialize_models might be needed.
    # Assuming models for feature extraction are not needed if `model` is a trained classifier
    # and data is already preprocessed or embeddings are loaded by a loader.
    # However, if `prepare_comic_dataset` and `precompute_embeddings` are called, models are needed.
    # For safety, let's ensure they are initialized if not already.
    global blip_processor
    if blip_processor is None: # A proxy to check if models were initialized
        print("Initializing supporting models for prediction's data preparation...")
        initialize_models(device) # device might be updated by initialize_models

    test_cache_dir = os.path.join(save_dir, "embeddings_cache", "test_full")
    os.makedirs(test_cache_dir, exist_ok=True)
    
    print(f"Loading {dataset_name} test dataset for task: {task}...")
    test_stream = load_dataset(dataset_name, task, split="test", streaming=True)
    
    print("Processing test dataset...")
    test_samples = prepare_comic_dataset(test_stream, "test", max_samples=None, ocr_df=ocr_df)
    print(f"Processed {len(test_samples)} test samples")
    
    print("Precomputing embeddings for test set...")
    test_processed = precompute_embeddings(test_samples, device, cache_dir=test_cache_dir)
    del test_samples; clear_gpu_memory()

    test_dataset = CachedEmbeddingsDataset(test_processed, cache_dir=test_cache_dir)
    del test_processed; clear_gpu_memory()
    print(f"Test dataset size: {len(test_dataset)}")
    
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, collate_fn=collate_fn, 
        num_workers=0, pin_memory=True
    )
    
    model.to(device).eval() # Ensure model is on correct device and in eval mode
    
    correct, total = 0, 0
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            try:
                context = batch["context"].to(device)
                options = batch["options"].to(device)
                labels = batch["label"].to(device)
                
                logits = model(context, options)
                _, predicted = torch.max(logits, 1)
                
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
            except Exception as e:
                print(f"Error during testing batch: {e}")
                clear_gpu_memory()
                continue
    
    accuracy = 100 * correct / total if total > 0 else 0
    print(f"Test accuracy: {accuracy:.2f}%")
    
    try:
        from sklearn.metrics import confusion_matrix, classification_report
        import json
        cm = confusion_matrix(all_labels, all_preds)
        print("Confusion Matrix:\n", cm)
        report = classification_report(all_labels, all_preds, zero_division=0)
        print("Classification Report:\n", report)
        
        results_path = os.path.join(save_dir, "test_results.json")
        results = {
            'accuracy': accuracy, 'confusion_matrix': cm.tolist(),
            'classification_report': report, 'predictions': all_preds,
            'true_labels': all_labels, 'correct_count': correct, 'total_count': total
        }
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=4)
        print(f"Test results saved to {results_path}")
    except ImportError:
        print("scikit-learn not available for detailed metrics. Please install it.")
    
    return accuracy, (all_preds, all_labels)


if __name__ == "__main__":
    # Set multiprocessing start method to 'spawn' for CUDA safety on some systems
    # This should be at the very beginning of the script execution.
    # However, placing it inside if __name__ == "__main__": is standard.
    try:
        mp.set_start_method('spawn', force=True)
        print("Multiprocessing start method set to 'spawn'.")
    except RuntimeError:
        print("Multiprocessing start method already set or failed to set to 'spawn'.")

    
    # --- Configuration ---
    # Common dataset settings
    DATASET_NAME = "VLR-CVC/ComicsPAP"
    TASK_NAME = "sequence_filling"
    
    # OCR dataframe (optional) - replace with your actual path
    OCR_DF_PATH = "/home/dbian/scripts/comics_train/final_ocr_data.pkl" # EXAMPLE PATH
    ocr_df = None
    if os.path.exists(OCR_DF_PATH):
        try:
            ocr_df = pd.read_pickle(OCR_DF_PATH)
            print(f"Loaded OCR data from {OCR_DF_PATH}")
            # Ensure 'sample_id' is string type if it's the index or used for merging
            if not isinstance(ocr_df.index, pd.Index) or ocr_df.index.dtype != 'object':
                 if 'sample_id' in ocr_df.columns:
                     ocr_df['sample_id'] = ocr_df['sample_id'].astype(str)
                     ocr_df = ocr_df.set_index('sample_id', drop=False) # Keep sample_id also as column if needed
                 else:
                     ocr_df.index = ocr_df.index.astype(str) # Assuming index is sample_id
            print("OCR DataFrame index type:", ocr_df.index.dtype)
            if 'split' not in ocr_df.columns:
                print(f"Warning: 'split' column not found in OCR DataFrame from {OCR_DF_PATH}. OCR filtering by split will not work.")


        except Exception as e:
            print(f"Could not load OCR data from {OCR_DF_PATH}: {e}")
            ocr_df = None
    else:
        print(f"OCR data file not found at {OCR_DF_PATH}. Proceeding without OCR filtering by split or external OCR data.")

    # Choose training mode by uncommenting one of the blocks below
    TRAINING_MODE = "full_dataset_chunks" # Options: "main_example", "chunks", "full_dataset", "full_dataset_chunks"
    RUN_PREDICTION = True

    # Common training settings (can be overridden in specific mode blocks)
    BATCH_SIZE = 16 # Adjusted from 20 due to potential OOM with larger models/data
    LEARNING_RATE = 5e-5 # Common starting point
    HIDDEN_DIM = 1024  # Adjusted from 1024
    SAVE_DIR_BASE = "comic_model_runs"
    
    model_trained = None
    history_data = None

    # --- Mode 4: `train_with_full_dataset_chunks` (HF 'train' in chunks, full HF 'val') ---
    if TRAINING_MODE == "full_dataset_chunks":
        print("\n--- Running in 'full_dataset_chunks' mode ---")
        CHUNK_SIZE_FULL_CHUNKS = 2000 # Process 1000 training samples at a time
        NUM_EPOCHS_PER_CHUNK_FULL_CHUNKS = 3 # Train for 5 epochs on each training chunk
        NUM_PASSES_FULL_CHUNKS = 3      # Go through entire training dataset twice
        LEARNING_RATE_FD_CHUNKS = 3e-4 # Potentially higher LR for longer training
        REPLAY_BUFFER_SIZE = 800 
        current_save_dir = os.path.join(SAVE_DIR_BASE, "full_dataset_chunks_run")
        
        model_trained, history_data = train_with_full_dataset_chunks(
            dataset_name=DATASET_NAME, task=TASK_NAME,
            batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE_FD_CHUNKS, hidden_dim=HIDDEN_DIM,
            chunk_size=CHUNK_SIZE_FULL_CHUNKS, num_epochs_per_chunk=NUM_EPOCHS_PER_CHUNK_FULL_CHUNKS,
            num_passes=NUM_PASSES_FULL_CHUNKS, save_dir=current_save_dir, ocr_df=ocr_df,
            replay_buffer_size=REPLAY_BUFFER_SIZE
        )
    else:
        print(f"Unknown TRAINING_MODE: {TRAINING_MODE}")
        current_save_dir = None


    # --- Prediction (optional, runs if a model was trained) ---
    if RUN_PREDICTION and model_trained and current_save_dir:
        print("\n--- Running Prediction on Test Set ---")
        
        # Option 1: Predict on test set (full, not chunked)
        test_accuracy, test_results = predict_on_test_set(
            model_trained, dataset_name=DATASET_NAME, task=TASK_NAME,
            batch_size=BATCH_SIZE, save_dir=current_save_dir, ocr_df=ocr_df
        )
        print(f"Final test accuracy (full test set): {test_accuracy:.2f}%")


    elif not model_trained and RUN_PREDICTION:
        print("Skipping prediction because no model was trained in this run.")
    elif not RUN_PREDICTION:
        print("Skipping prediction as RUN_PREDICTION is False.")

    print("\n--- Script execution finished ---")

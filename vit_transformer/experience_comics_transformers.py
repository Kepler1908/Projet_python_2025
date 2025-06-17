# -*- coding: utf-8 -*-
from torchvision import transforms
from PIL import Image
from torchvision.transforms import InterpolationMode

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vit_b_32, ViT_B_32_Weights


from torchvision.transforms import PILToTensor
from torch.optim import Adam
from tqdm import tqdm 

# Définition des transformations pour prétraiter les images
transform = transforms.Compose([
    transforms.Resize(256, interpolation=InterpolationMode.BILINEAR),  # Redimensionne l'image à 256 pixels
    transforms.CenterCrop(224),  # Recadre l'image au centre à 224 pixels
    transforms.ToTensor(),  # Convertit l'image en tenseur
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Normalise les valeurs des pixels
                         std=[0.229, 0.224, 0.225])
])

# Création d'une image noire pour le padding
black_image = transform(Image.new("RGB", (224, 224), (0, 0, 0)))

# Fonction pour prétraiter un échantillon
# Cette fonction applique les transformations définies ci-dessus aux images du contexte et des options
# Elle ajoute également un padding pour garantir que les options contiennent toujours 5 images
# Enfin, elle génère les positions des contextes en excluant l'indice de la solution

def preprocess_sample(sample):
    # Applica le trasformazioni
    context_imgs = [transform(img) for img in sample['context']]
    options_imgs = [transform(img) for img in sample['options']]

    # Padding per portare options a 5 immagini
    while len(options_imgs) < 5:
        options_imgs.append(black_image)

    sample['context'] = torch.stack(context_imgs)
    sample['options'] = torch.stack(options_imgs)

    list_indices = [0, 1, 2, 3, 4]
    sample['context_positions'] = torch.tensor([e for e in list_indices if e != sample['index']])

    return sample

# Chargement du dataset ComicsPAP avec le skill et le split spécifiés
skill = "sequence_filling"  # Type de tâche à résoudre
split = "train"  # Partie du dataset à utiliser

# Chargement et prétraitement du dataset
from datasets import load_dataset

dataset = load_dataset("VLR-CVC/ComicsPAP", skill, split=split, streaming=True)
dataset = dataset.map(preprocess_sample)

batch_train = 16  # Taille des batch pour l'entraînement
batched_dataset = dataset.batch(batch_train)

# Définition du modèle ComicClozeModel
# Ce modèle utilise un Vision Transformer (ViT-B/32) pré-entraîné pour extraire des embeddings visuels
# Il utilise également un encodeur Transformer pour agréger les informations contextuelles
class ComicClozeModel(nn.Module):
    def __init__(self, embedding_dim=768, transformer_layers=1, num_positions=5):
        super().__init__()

        # Backbone Vision Transformer (pre-trained ViT-B/32)
        self.backbone = vit_b_32(weights=ViT_B_32_Weights.IMAGENET1K_V1)
        self.backbone.heads = nn.Identity()  # Supprime la tête de classification
        backbone_output_dim = 768  # Dimension des embeddings du ViT-B/32

        self.pos_embedding = nn.Embedding(num_positions, embedding_dim)  # Embedding pour les positions des options

        # Transformer aggregator
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim, nhead=8, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=transformer_layers
        )

    def forward(self, context_panels, context_positions, candidate_panels):
        """
        context_panels: Tensor (B, N_context, 3, H, W)
        context_positions: Tensor (B, N_context) valeurs entre 0 et num_positions-1
        candidate_panels: Tensor (B, N_candidates, 3, H, W)
        """
        B, N_context, C, H, W, = context_panels.shape
        N_candidates = candidate_panels.shape[1]

        # Aplatissement des images de contexte et des candidats pour un traitement par batch
        context_panels_flat = context_panels.reshape(-1, 3, H, W)
        candidate_panels_flat = candidate_panels.reshape(-1, 3, H, W)

        # Extraction des embeddings visuels
        context_emb = self.backbone(context_panels_flat) # (B*N_context, embedding_dim)
        candidate_emb = self.backbone(candidate_panels_flat)

        # Reshape pour revenir à (B, N, embedding_dim)
        context_emb = context_emb.view(B, N_context, -1)
        candidate_emb = candidate_emb.view(B, N_candidates, -1)

        # Ajout des embeddings de position
        pos_emb = self.pos_embedding(context_positions)  # (B, N_context, embedding_dim)
        context_emb = context_emb + pos_emb

        # Agrégation avec le Transformer
        aggregated_context = self.transformer(context_emb)  # (B, N_context, embedding_dim)

        # Moyenne des embeddings de contexte
        aggregated_context = aggregated_context.mean(dim=1)  # (B, embedding_dim)

        # Calcul des scores par produit scalaire avec les candidats
        scores = torch.bmm(candidate_emb, aggregated_context.unsqueeze(-1)).squeeze(-1)  # (B, N_candidates)

        return scores


# Initialisation du modèle
model = ComicClozeModel()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Définition de l'optimiseur et du critère
optimizer = Adam(model.parameters(), lr=3e-4)
criterion = nn.CrossEntropyLoss()

# Chargement du dataset de test
split = "val"  # Partie du dataset à utiliser pour le test
dataset_test = load_dataset("VLR-CVC/ComicsPAP", skill, split=split, streaming=True)

# Prétraitement du dataset de test
dataset_test = dataset_test.map(preprocess_sample)

batch_test = 16  # Taille des batch pour le test
batched_dataset_test = dataset_test.batch(batch_test)

# Fonction pour tester le modèle
# Cette fonction évalue le modèle sur le dataset de test et calcule la perte et l'exactitude

def test_model():
  model.eval()  
  test_loss = 0
  correct_test = 0
  total_test = 0

  total_1 = 0
  total_step = 0

  for batch in tqdm(batched_dataset_test):
      total_step += 1
      context_panels = torch.stack(batch['context']).to(device)
      context_positions = torch.stack(batch['context_positions']).to(device)
      candidate_panels = torch.stack(batch['options']).to(device)
      labels = torch.tensor(batch['solution_index']).to(device)

      if labels[0].item()==-1:
        labels[0] = candidate_panels.shape[1] - 1

      # Passage avant (forward pass)
      output = model(context_panels, context_positions, candidate_panels)

      # Calcul de la perte (loss)
      loss = criterion(output, labels)
      test_loss += loss.item()

      # Rétropropagation et optimisation
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      # Calcul de l'exactitude
      preds = torch.argmax(output, dim=1)
      correct_test += (preds == labels).sum().item()
      total_test += labels.size(0)

  avg_loss = test_loss / total_step
  accuracy = correct_test / total_test * 100

  print(f"Accuracy: {accuracy:.2f}% | Loss: {avg_loss:.4f}\n")

# Paramètres de training
num_epochs = 10

# Boucle d'entraînement
# Cette boucle entraîne le modèle sur le dataset d'entraînement et teste le modèle à chaque époque
for epoch in range(num_epochs):

    if epoch % 1 == 0:
      print('Testing model')
      test_model()
      print('End test. Resuming training')

    model.train()  
    epoch_loss = 0
    correct = 0
    total = 0

    item_count = 0

    for batch in tqdm(batched_dataset, desc=f"Epoch {epoch+1}/{num_epochs}"):
        item_count += 1
        # Sposta i dati su GPU se disponibile
        context_panels = torch.stack(batch['context']).to(device)
        context_positions = torch.stack(batch['context_positions']).to(device)
        candidate_panels = torch.stack(batch['options']).to(device)
        labels = torch.tensor(batch['solution_index']).to(device)

        # Forward pass
        output = model(context_panels, context_positions, candidate_panels)

        # Calcolo della loss
        loss = criterion(output, labels)
        epoch_loss += loss.item()

        # Backprop e ottimizzazione
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Calcolo accuratezza
        preds = torch.argmax(output, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    avg_loss = epoch_loss / item_count
    accuracy = correct / total * 100

    print(f"\nEpoch {epoch+1} | Accuracy: {accuracy:.2f}% | Loss: {avg_loss:.4f}\n")
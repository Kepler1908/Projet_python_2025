# Projet_python_2025

Ce projet Python explore la reconnaissance de séquences narratives dans les bandes dessinées, une tâche multimodale complexe nécessitant l’analyse conjointe d’éléments visuels et textuels hétérogènes. À partir du jeu de données ComicsPAP, conçu pour l’évaluation fine de modèles d’intelligence artificielle sur des BD de styles variés, nous expérimentons différentes approches pour prédire l’ordre narratif des vignettes.

## Structure
- `vit_transformer`/`experience_comics_transformers.py`:

    Contient le code pour l'entrainement du modèle décrit ci-dessous et dans le paragraph _ViT-B/32 + Transformer Layer_ du rapport de projet.
    - **Vision Transformer (ViT-B/32)** : Utilisé pour extraire des représentations visuelles des panneaux de bandes dessinées
    - **Enrichissement des représentations** : Ajout d'embeddings positionnels aux embeddings des panneaux de contexte
    - **Agrégation des contextes** via un encodeur Transformer
    - **Évaluation des candidats** : Calcul des scores basés sur le produit scalaire entre les contextes agrégés et les panneaux candidats
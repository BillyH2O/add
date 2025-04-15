# ADD0 - Adversarial Diffusion Distillation pour la génération d'images Pokémon

<p align="center">
  <img src="https://github.com/user-attachments/assets/e95e3396-5087-4fbb-ade3-9fc033a62097" alt="dielkiatieus">
</p>



Ce débot contient une implémentation de la méthode Adversarial Diffusion Distillation (ADD) pour générer des images de Pokémon de haute qualité en un minimum d'étapes. Nous étudions l'efficacité de cette approche à travers trois configurations :

Un modèle teacher basé sur Stable Diffusion v1-4, utilisé sans fine-tuning (configuration de base).
Le même modèle teacher après un fine-tuning sur deux datasets Pokémon de tailles différentes (809 et 11 325 images).
Un modèle student entraîné avec ADD, combinant distillation et apprentissage adversariale, sur ces mêmes datasets.

## Fonctionnalités

- Génération d'un dataset Pokemon 
- Pré-traitement de ce dataset
- Fine-tuning de Stable Diffusion v1-4 sur des datasets Pokémon personnalisés
- Implémentation d'ADD pour une inférence efficace (1 à 4 étapes)
- Interface web interactive pour la génération d'images

## Structure du projet

```
├── docs/                   # Documentation
│   ├── rapport_add0.pdf    # rapport pdf 
├── data/                   # Données brutes et prétraitées (non incluses)
├── models/                 # Modèles pré-entraînés (non inclus)
├── les notebooks          
│   ├── add.ipynb           # Démonstration de l'implémentation ADD
│   ├── interface.ipynb     # Interface interactive pour la génération
│   ├── eda.ipynb           # Analyse exploratoire des données
│   ├── finetuning.ipynb    # Fine-tuning sur dataset n°1
│   ├── finetuning12k.ipynb # Fine-tuning sur dataset n°2
│   └── preprocessing.ipynb # Prétraitement des données
├── les scripts             
│   ├── modelADD.py         # Définition du modèle ADD
│   ├── trainADD.py         # Script d'entraînement du modèle ADD
│   ├── generateADD.py      # Script de génération d'images
│   ├── preprocessing.py    # Prétraitement pour dataset n°1
│   ├── preprocessing12k.py # Prétraitement pour dataset n°2
│   ├── interface.py        # Interface utilisateur pour la génération
│   └── test_quality.py     # Évaluation de la qualité des images
├── prompts.txt             # Exemples de prompts pour la génération d'images
├── requirements.txt        # Dépendances du projet
└── README.md               # Ce fichier
```

## Installation

1. Clonez le dépôt :
```bash
git clone https://github.com/BillyH20/add0.git
cd add
```

2. Installez les dépendances :
```bash
pip install -r requirements.txt
```

3. Téléchargez les datasets Pokémon depuis les sources suivantes :
   - [Pokemon Image Dataset](https://www.kaggle.com/datasets/hlrhegemony/pokemon-image-dataset)
   - [Pokemon Images Dataset](https://www.kaggle.com/datasets/kvpratama/pokemon-images-dataset)
   - [Pokemon Images and Types](https://www.kaggle.com/datasets/vishalsubbiah/pokemon-images-and-types)
   - [Complete Pokemon Dataset](https://www.kaggle.com/datasets/mariotormo/complete-pokemon-dataset-updated-090420)

   Ces datasets contiennent plus de 2,500 images proprement étiquetées, toutes des illustrations officielles, pour les générations 1 à 8.

4. Exécutez le script `generate_dataset.py` pour générer le répertoire pour les images et annotations (une clé OPENAI sera nécessaire) :
```bash
python generate_dataset.py
```
Ce script générera le dataset sur le chemin courant. La taille du dataset final dépendra du nombre de datasets Kaggle que vous aurez téléchargés. Il ne vous restera plus qu'à déplacer ces datasets dans le répertoire `/data/pokemon_images`.

## Utilisation

### 1. Prétraitement des données

Prétraitez les datasets Pokémon pour les rendre compatibles avec l'entraînement :

```bash
jupyter notebook preprocessing.ipynb
```

ou bien

```bash
python preprocessing.py    # Pour dataset n°1 (809 images)
python preprocessing12k.py # Pour dataset n°2 (11325 images)
```

### 2. Entraînement du modèle

Entraînez le modèle ADD ou fine-tunez Stable Diffusion v1-4 :

```bash
jupyter notebook add.ipynb
```

ou bien

```bash
python trainADD.py
```

Modifiez les hyperparamètres dans trainADD.py si nécessaire (voir rapport pour les détails).

Pour fine-tuner Stable Diffusion v1-4 sur le deuxième dataset, lancez le notebook `finetuning.ipynb` :
```bash
jupyter notebook finetuning.ipynb
```

### 3. Génération d'images

Générez des images à partir de prompts textuels :

```bash
python generateADD.py --prompt "A cute Pikachu with red cheeks playing in a grassy field"
```

Options disponibles : `--num_inference_steps`, `--guidance_scale`, etc.

### 4. Interface utilisateur

Lancez l'interface interactive pour générer des images en temps réel :

```bash
streamlit run interface.py
```

Accédez à l'interface via votre navigateur (par défaut : http://localhost:8501).

ou bien si vous êtes sur google colab 

```bash
jupyter notebook interface.ipynb
```

## Notebooks

Les notebooks Jupyter offrent une exploration interactive :

- `add.ipynb` : Testez le modèle ADD étape par étape
- `interface.ipynb` : Expérimentez avec l'interface de génération
- `eda.ipynb` : Visualisez et analysez les datasets Pokémon
- `finetuning.ipynb` / `finetuning12k.ipynb` : Reproduisez le fine-tuning
- `preprocessing.ipynb` : Comprenez le pipeline de prétraitement

## Références

- Sauer, A., et al. Adversarial Diffusion Distillation : https://arxiv.org/abs/2311.17042
- Stable Diffusion v1-4 : [Hugging Face](https://huggingface.co/CompVis/stable-diffusion-v1-4)
- Datasets Pokémon : [Kaggle](https://www.kaggle.com/datasets)
- Code source : [GitHub ](https://github.com/BillyH20/add)

## Remarques

- L'entraînement ADD peut être instable sans un réglage fin des hyperparamètres (voir section 7.5 du rapport).
- Pour de meilleurs résultats, utilisez un dataset diversifié (recommandé : dataset n°2 avec 11325 images).
- Consultez le rapport (docs/rapport_add0.pdf) pour une analyse détaillée des performances.

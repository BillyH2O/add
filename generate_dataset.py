import os
import re
import json
import shutil
import pandas as pd
import requests
from PIL import Image
from typing import Dict
from openai import OpenAI

# Étape 1 : Créer le fichier pokedex.json avec PokeAPI
pokedex_json_path = "pokedex.json"
pokedex = {}

# Récupérer les Pokémon de 1 à 809 via PokeAPI
for i in range(1, 810):  # 1 à 809 inclus
    response = requests.get(f"https://pokeapi.co/api/v2/pokemon/{i}")
    if response.status_code == 200:
        data = response.json()
        # Format : {"pikachu": "25"} (nom en minuscules, numéro comme valeur)
        pokedex[data["name"].lower()] = str(i)
    else:
        print(f"Erreur pour le Pokémon #{i}")

# Sauvegarder le fichier pokedex.json
with open(pokedex_json_path, "w", encoding="utf-8") as f:
    json.dump(pokedex, f, indent=4)
print(f"✅ Fichier {pokedex_json_path} créé avec succès !")

# Étape 2 : Organiser les images dans le dossier images en utilisant pokedex.json
images_folder = "images"

# Charger le dictionnaire Pokédex
try:
    with open(pokedex_json_path, "r", encoding="utf-8") as f:
        pokedex_dict: Dict[str, str] = json.load(f)
except FileNotFoundError:
    print(f"Erreur : Le fichier {pokedex_json_path} n'existe pas.")
    exit(1)

# Créer un dictionnaire inversé pour recherche rapide
number_to_name = {str(v): k for k, v in pokedex_dict.items()}
print(f"Nombre de Pokémon dans le JSON : {len(pokedex_dict)}")
print(f"Exemple de correspondances : 25={number_to_name.get('25')}, 1={number_to_name.get('1')}")

def extract_pokedex_number(filename: str) -> str | None:
    """Extrait le numéro Pokédex du début du nom de fichier."""
    match = re.match(r"(\d+)", filename.lower())
    return match.group(1) if match else None

def process_image_file(filename: str, source_folder: str, pokedex: Dict[str, str]) -> None:
    """Renomme et déplace un fichier image dans un sous-dossier basé sur le nom du Pokémon."""
    if not filename.lower().endswith((".jpg", ".png")):
        print(f"Ignoré (pas un JPG/PNG) : {filename}")
        return

    pokedex_num = extract_pokedex_number(filename)
    if not pokedex_num:
        print(f"Pas de numéro Pokédex détecté : {filename}")
        return

    if pokedex_num not in pokedex:
        print(f"Pas de correspondance Pokédex pour {pokedex_num} dans {filename}")
        return

    pokemon_name = pokedex[pokedex_num]
    suffix = filename[len(pokedex_num):].lstrip("-_") or ".jpg"
    new_filename = f"{pokemon_name}{suffix}"

    # Créer le sous-dossier
    subfolder = os.path.join(source_folder, pokemon_name)
    os.makedirs(subfolder, exist_ok=True)

    # Déplacer le fichier
    old_path = os.path.join(source_folder, filename)
    new_path = os.path.join(subfolder, new_filename)
    try:
        shutil.move(old_path, new_path)
        print(f"✔ {filename} => {pokemon_name}/{new_filename}")
    except Exception as e:
        print(f"Erreur lors du déplacement de {filename} : {e}")

# Traitement des fichiers dans images
for fname in os.listdir(images_folder):
    process_image_file(fname, images_folder, number_to_name)

print("✅ Organisation des images dans 'images' terminée !")

# Étape 3 : Transférer les images de images_2 vers images
source_folder = "images_2"
destination_folder = "images"

# Vérifier que les dossiers existent
if not os.path.exists(source_folder):
    print(f"Erreur : Le dossier {source_folder} n'existe pas.")
    exit(1)
if not os.path.exists(destination_folder):
    print(f"Erreur : Le dossier {destination_folder} n'existe pas.")
    exit(1)

# Parcourir les sous-dossiers dans images_2
for subfolder in os.listdir(source_folder):
    source_subfolder_path = os.path.join(source_folder, subfolder)
    
    if not os.path.isdir(source_subfolder_path):
        print(f"Ignoré (pas un dossier) : {subfolder}")
        continue
    
    # Convertir le nom du sous-dossier en minuscules pour correspondre à images
    pokemon_name = subfolder.lower()
    dest_subfolder_path = os.path.join(destination_folder, pokemon_name)
    
    # Créer le dossier de destination si nécessaire
    if not os.path.exists(dest_subfolder_path):
        print(f"Création du dossier : {dest_subfolder_path}")
        os.makedirs(dest_subfolder_path)
    
    # Parcourir les fichiers dans le sous-dossier source
    for filename in os.listdir(source_subfolder_path):
        source_file_path = os.path.join(source_subfolder_path, filename)
        
        if not os.path.isfile(source_file_path):
            print(f"Ignoré (pas un fichier) : {filename}")
            continue
        
        # Chemin de destination
        dest_file_path = os.path.join(dest_subfolder_path, filename)
        
        # Vérifier si le fichier existe déjà
        if os.path.exists(dest_file_path):
            print(f"Fichier déjà existant, ignoré : {dest_file_path}")
            continue
        
        # Transférer (déplacer) le fichier
        try:
            shutil.move(source_file_path, dest_file_path)
            print(f"✔ {source_file_path} => {dest_file_path}")
        except Exception as e:
            print(f"Erreur lors du transfert de {source_file_path} : {e}")

print("✅ Transfert de 'images_2' vers 'images' terminé !")

# Étape 4 : Créer pokemon.xlsx à partir de pokemon.csv et des fichiers de species
# Charger le fichier principal
csv_file = "pokemon.csv"
df = pd.read_csv(csv_file)

# Charger et fusionner les 3 fichiers contenant les species
species_files = [
    "pokedex_(Update.04.20).csv",
    "pokedex_(Update_04.21).csv",
    "pokedex_(Update_05.20).csv"
]

# Fusionner les fichiers contenant les species
species_df_list = [pd.read_csv(f) for f in species_files]
species_df = pd.concat(species_df_list, ignore_index=True)

# Garder uniquement les colonnes utiles et éviter les doublons
species_df = species_df[['name', 'species']].dropna().drop_duplicates()
species_df['name'] = species_df['name'].str.lower()

# Chemin vers les images
image_folder = "images"

# Traitement fusionné
data_with_paths = []

for idx, row in df.iterrows():
    name = row["Name"]
    type1 = row["Type1"]
    type2 = row["Type2"]

    # Formatage du nom pour chercher un dossier et faire le merge
    name_lower = name.lower()

    # Chercher le dossier (et non une image individuelle)
    image_path = os.path.join(image_folder, name_lower)
    image_value = image_path if os.path.exists(image_path) else "No image"

    # Chercher la species
    species_match = species_df[species_df["name"] == name_lower]
    if not species_match.empty:
        species = species_match.iloc[0]["species"]
    else:
        species = "Unknown"

    data_with_paths.append({
        "Name": name,
        "Type1": type1,
        "Type2": type2,
        "Species": species,
        "ImagePath": image_value
    })

# Sauvegarde Excel
df_out = pd.DataFrame(data_with_paths)
output_file = "pokemon.xlsx"
df_out.to_excel(output_file, index=False)
print(f"Fichier Excel sauvegardé sous {output_file}")

# Étape 5 : Générer les descriptions avec OpenAI
openai_client = OpenAI(
    api_key="your-key"
)

def generate_openai_desc(row):
    name = row["Name"]
    types = f"{row['Type1']}" + (f" and {row['Type2']}" if pd.notna(row['Type2']) and row['Type2'] != "" else "")
    species = row["Species"]
    
    prompt = (f"Describe in a very short and creative description {name}, a {types.lower()} type Pokémon, and a {species.lower()}. "
              f"Focus on its appearance, not its abilities or powers. "
              f"Limit the description to one sentence.")
    
    response = openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5,
        max_tokens=50
    )

    return response.choices[0].message.content.strip()

# Charger le fichier Excel
df_out = pd.read_excel("pokemon.xlsx")

# Générer la colonne Description avec OpenAI
descriptions = []
for idx, row in df_out.iterrows():
    try:
        description = generate_openai_desc(row)
        print(f"{row['Name']} ➜ {description}")
    except Exception as e:
        print(f"Erreur à la ligne {idx} ({row['Name']}) : {e}")
        description = "Description unavailable"
    descriptions.append(description)

df_out["Description"] = descriptions

# Sauvegarder dans un nouveau fichier Excel
output_file = "pokemon.xlsx"
df_out.to_excel(output_file, index=False)
print(f"✅ Fichier Excel enrichi sauvegardé sous : {output_file}")

# Étape 6 : Créer le fichier labels.csv
# Charger le fichier Excel
excel_file = "pokemon.xlsx"
df_pokemon = pd.read_excel(excel_file)

# Convertir les noms en minuscules pour correspondre aux dossiers
df_pokemon['Name'] = df_pokemon['Name'].str.lower()

# Créer un dictionnaire pour mapper les noms aux métadonnées
pokemon_info = df_pokemon.set_index('Name')[['Type1', 'Type2', 'Species', 'Description', 'ImagePath']].to_dict('index')

# Liste pour stocker les données
data = []

# Fonction pour convertir une image en JPG
def convert_to_jpg(image_path, output_path):
    try:
        img = Image.open(image_path).convert("RGB")
        img.save(output_path, "JPEG", quality=95)
        print(f"✔ Converti en JPG : {image_path} -> {output_path}")
        return True
    except Exception as e:
        print(f"Erreur lors de la conversion de {image_path} : {e}")
        return False

# Parcourir chaque Pokémon
for pokemon_name, info in pokemon_info.items():
    type1 = info['Type1']
    type2 = info['Type2']
    species = info['Species']
    description = info['Description']
    folder_path = info['ImagePath']
    
    if not os.path.isdir(folder_path):
        print(f"⚠ Dossier non trouvé pour {pokemon_name} : {folder_path}")
        continue
    
    # Parcourir les images dans le dossier
    for filename in os.listdir(folder_path):
        image_path = os.path.join(folder_path, filename)
        
        if not os.path.isfile(image_path):
            print(f"Ignoré (pas un fichier) : {image_path}")
            continue
        
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            # Convertir en JPG si nécessaire
            if not filename.lower().endswith('.jpg'):
                new_filename = os.path.splitext(filename)[0] + ".jpg"
                new_image_path = os.path.join(folder_path, new_filename)
                if convert_to_jpg(image_path, new_image_path):
                    os.remove(image_path)
                    image_path = new_image_path
                else:
                    continue
            # Ajouter au dataset
            data.append({
                'Name': pokemon_name,
                'ImagePath': image_path,
                'Type1': type1,
                'Type2': type2,
                'Species': species,
                'Description': description,
            })
        else:
            print(f"Ignoré (pas une image PNG/JPG) : {image_path}")

# Créer un DataFrame et sauvegarder en CSV
df = pd.DataFrame(data)
csv_file = "labels.csv"
df.to_csv(csv_file, index=False)

print(f"✅ Fichier CSV généré : {csv_file}")

# Lecture du CSV
df = pd.read_csv(csv_file)

# Export en Excel
output_file = "labels.xlsx"
df.to_excel(output_file, index=False)
print(f"Fichier Excel généré : {output_file}")
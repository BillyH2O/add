import os
import pandas as pd
from PIL import Image
import torch
from torchvision import transforms
from tqdm import tqdm
import argparse


def prepare_pokemon_dataset(platform="windows", base_dir="data", output_csv="pokemon_train.csv"):
    """
    Prépare le dataset local Pokémon (1k) avec une gestion multiplateforme.
    """
    dataset_path = os.path.join(base_dir, "pokemon_images")
    annotations_path = os.path.join(dataset_path, "annotations.xlsx")
    output_dir = os.path.join(dataset_path, "processed")

    print(f"Chargement des annotations depuis : {annotations_path}")
    df = pd.read_excel(annotations_path, engine='openpyxl')

    df.columns = ['name', 'type1', 'type2', 'species', 'imagePath', 'description']

    df['type'] = df.apply(
        lambda row: row['type1'] + (f"/{row['type2']}" if pd.notna(row['type2']) and row['type2'] != '' else ''),
        axis=1
    )

    df['caption'] = df.apply(
        lambda row: f"{row['name']}, a {row['type']} type {row['species']}. {row['description']}",
        axis=1
    )

    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(os.path.join(output_dir, "pokemon_metadata.csv"), index=False)
    print(f"Metadata sauvegardée dans {os.path.join(output_dir, 'pokemon_metadata.csv')}")

    transform = transforms.Compose([
        transforms.Resize(512),
        transforms.CenterCrop(512),
        transforms.ToTensor()
    ])

    image_paths, captions, types, names = [], [], [], []
    found_images, not_found_images = 0, 0

    for i, row in tqdm(df.iterrows(), total=len(df), desc="Prétraitement des images"):
        try:
            image_path = row['imagePath']
            if platform.lower() == "linux":
                image_path = image_path.replace("\\", "/")

            possible_paths = [
                os.path.join(dataset_path, image_path),
                os.path.join(dataset_path, os.path.basename(image_path)),
                os.path.join(dataset_path, "images", os.path.basename(image_path))
            ]

            img_path = next((p for p in possible_paths if os.path.exists(p)), None)

            if img_path is None:
                not_found_images += 1
                if not_found_images <= 5:
                    print(f"Image introuvable : {image_path} | chemins essayés : {possible_paths}")
                continue

            image = Image.open(img_path).convert("RGB")
            transformed = transform(image)

            save_img_path = os.path.join(output_dir, f"pokemon_{i:04d}.png")
            torch_path = os.path.join(output_dir, f"pokemon_{i:04d}.pt")

            image.save(save_img_path)
            torch.save(transformed, torch_path)

            image_paths.append(save_img_path)
            captions.append(row["caption"])
            types.append(row["type"])
            names.append(row["name"])

            if found_images < 3:
                print(f"{row['name']} → {save_img_path}")

            found_images += 1

        except Exception as e:
            print(f"Erreur lors du traitement de {row['name']}: {e}")
            continue

    print(f"\nRésumé : {found_images} images traitées, {not_found_images} manquantes")

    processed_df = pd.DataFrame({
        "name": names,
        "image_path": image_paths,
        "type": types,
        "caption": captions
    })

    final_csv_path = os.path.join(base_dir, output_csv)
    processed_df.to_csv(final_csv_path, index=False)
    print(f"Dataset final sauvegardé : {final_csv_path}")
    return processed_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Préparation du dataset Pokémon 1k (multi-plateforme)")
    parser.add_argument("--platform", type=str, choices=["windows", "linux"], default="windows",
                        help="Plateforme (windows ou linux/colab)")
    parser.add_argument("--base_dir", type=str, default="data", help="Répertoire racine du dataset")
    parser.add_argument("--output_csv", type=str, default="pokemon_train.csv", help="Nom du fichier de sortie CSV")
    args = parser.parse_args()

    prepare_pokemon_dataset(platform=args.platform, base_dir=args.base_dir, output_csv=args.output_csv)

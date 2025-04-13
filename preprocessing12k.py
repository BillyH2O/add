import os
import pandas as pd
from PIL import Image
import torch
from torchvision import transforms
from tqdm import tqdm
import argparse

def prepare_12k_pokemon_dataset(platform="windows", base_dir="data", output_csv="pokemon_train_12k.csv"):
    """
    Prépare le dataset 12k avec support multiplateforme.
    Structure : <base_dir>/pokemon_images/images12k/<pokemon_name>/<image>.jpg
    Annotations : <base_dir>/pokemon_images/annotations12k.xlsx
    """
    dataset_path = os.path.join(base_dir, "pokemon_images")
    images_root = os.path.join(dataset_path, "images12k")
    annotations_path = os.path.join(dataset_path, "annotations12k.xlsx")
    output_dir = os.path.join(dataset_path, "processed_12k")

    print(f"Chargement des annotations depuis : {annotations_path}")
    df = pd.read_excel(annotations_path, engine="openpyxl")

    df.columns = ["name", "type1", "type2", "species", "imagePath", "description"]

    df["type"] = df.apply(
        lambda row: row["type1"] + (f"/{row['type2']}" if pd.notna(row["type2"]) and row["type2"] != "" else ""),
        axis=1
    )

    df["caption"] = df.apply(
        lambda row: f"{row['name']}, a {row['type']} type {row['species']}. {row['description']}",
        axis=1
    )

    os.makedirs(output_dir, exist_ok=True)
    metadata_path = os.path.join(output_dir, "pokemon_metadata_12k.csv")
    df.to_csv(metadata_path, index=False)
    print(f"Métadonnées sauvegardées dans : {metadata_path}")

    transform = transforms.Compose([
        transforms.Resize(512),
        transforms.CenterCrop(512),
        transforms.ToTensor()
    ])

    image_paths, captions, types, names = [], [], [], []
    found_images, not_found_images = 0, 0

    for i, row in tqdm(df.iterrows(), total=len(df), desc="Prétraitement des images"):
        try:
            image_path = row["imagePath"]
            if platform.lower() == "linux":
                image_path = image_path.replace("\\", "/")
            if image_path.startswith("images/"):
                image_path = image_path[len("images/"):]

            full_path = os.path.join(images_root, image_path)

            if not os.path.exists(full_path):
                not_found_images += 1
                if not_found_images <= 5:
                    print(f"Image introuvable : {full_path}")
                continue

            image = Image.open(full_path).convert("RGB")
            transformed = transform(image)

            save_img_path = os.path.join(output_dir, f"pokemon12k_{i:05d}.png")
            torch_path = os.path.join(output_dir, f"pokemon12k_{i:05d}.pt")

            image.save(save_img_path)
            torch.save(transformed, torch_path)

            image_paths.append(save_img_path)
            #relative_img_path = os.path.relpath(save_img_path, base_dir).replace("\\", "/")
            #image_paths.append(relative_img_path)

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

    final_csv_path = os.path.join(base_dir, output_csv)
    processed_df = pd.DataFrame({
        "name": names,
        "image_path": image_paths,
        "type": types,
        "caption": captions
    })
    processed_df.to_csv(final_csv_path, index=False)
    print(f"Dataset final sauvegardé : {final_csv_path}")
    return processed_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Préparation du dataset 12k Pokémon (multi-plateforme)")
    parser.add_argument("--platform", type=str, choices=["windows", "linux"], default="windows",
                        help="Plateforme (windows ou linux/colab)")
    parser.add_argument("--base_dir", type=str, default="data",
                        help="Répertoire de base des données")
    parser.add_argument("--output_csv", type=str, default="pokemon_train_12k.csv",
                        help="Chemin relatif du CSV de sortie")
    args = parser.parse_args()

    prepare_12k_pokemon_dataset(platform=args.platform, base_dir=args.base_dir, output_csv=args.output_csv)

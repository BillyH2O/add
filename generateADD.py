import os
import argparse
import torch
from PIL import Image
import numpy as np
from modelADD import PokemonADD
from tqdm import tqdm
import time

def main(args):
    """Génère des images Pokémon à partir du modèle ADD entraîné"""
    # Configuration du dispositif
    device = torch.device(args.device)
    print(f"Utilisation du dispositif: {device}")
    
    # Création du répertoire de sortie
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Chargement du modèle
    print(f"Chargement du modèle ADD depuis {args.model_path}...")
    add_model = PokemonADD(
        pretrained_model_name_or_path=args.pretrained_model,
        enable_xformers=args.enable_xformers,
        device=device
    )
    
    # Chargement des poids sauvegardés de l'étudiant
    if os.path.exists(args.model_path):
        checkpoint = torch.load(args.model_path, map_location=device)
        if 'student_state_dict' in checkpoint:
            add_model.student.load_state_dict(checkpoint['student_state_dict'])
            print("Chargement réussi des poids du modèle étudiant")
        else:
            print("Impossible de trouver les poids de l'étudiant dans le point de contrôle")
    else:
        print(f"Avertissement: Le chemin du modèle {args.model_path} n'existe pas. Utilisation des poids préentraînés.")
    
    # Lecture des prompts depuis un fichier si fourni
    if args.prompt_file:
        with open(args.prompt_file, 'r') as f:
            prompts = [line.strip() for line in f.readlines() if line.strip()]
    else:
        prompts = args.prompts if args.prompts else [
            "a cute Pikachu with red cheeks",
            "a powerful Charizard breathing fire",
            "a cheerful Squirtle with a blue shell",
            "a happy Eevee with fluffy fur"
        ]
    
    print(f"Génération de {len(prompts)} images...")
    
    # Passage du modèle en mode évaluation
    add_model.eval()
    
    # Génération d'images pour chaque nombre d'étapes d'inférence
    for num_steps in args.num_inference_steps:
        print(f"\nGénération avec {num_steps} étape(s) d'inférence...")
        
        # Création du répertoire pour ce nombre d'étapes
        step_dir = os.path.join(args.output_dir, f"{num_steps}step")
        os.makedirs(step_dir, exist_ok=True)
        
        # Chronométrage de la génération
        start_time = time.time()
        
        # Génération des images
        with torch.no_grad():
            for i, prompt in enumerate(tqdm(prompts)):
                # Génération de l'image
                images = add_model.generate(
                    [prompt],
                    num_inference_steps=num_steps,
                    guidance_scale=args.guidance_scale,
                    height=args.height,
                    width=args.width
                )
                
                # Sauvegarde de l'image
                for j, img in enumerate(images):
                    img_pil = Image.fromarray(img)
                    
                    # Création d'un nom de fichier sécurisé
                    safe_prompt = "".join([c if c.isalnum() else "_" for c in prompt])[:50]
                    filename = f"{i:03d}_{safe_prompt}.png"
                    img_path = os.path.join(step_dir, filename)
                    
                    img_pil.save(img_path)
        
        # Calcul et affichage du temps de génération
        end_time = time.time()
        elapsed_time = end_time - start_time
        time_per_image = elapsed_time / len(prompts)
        
        print(f"Génération de {len(prompts)} images en {elapsed_time:.2f} secondes")
        print(f"Temps moyen par image: {time_per_image:.3f} secondes")
        print(f"Images sauvegardées dans {step_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Génération d'images Pokémon avec le modèle ADD")
    
    # Arguments du modèle
    parser.add_argument("--model_path", type=str, default="output/models/add_model_final.pt",
                        help="Chemin vers le point de contrôle du modèle entraîné")
    parser.add_argument("--pretrained_model", type=str, default="CompVis/stable-diffusion-v1-4",
                        help="Modèle préentraîné à initialiser")
    parser.add_argument("--enable_xformers", action="store_true",
                        help="Utiliser xformers pour une attention économe en mémoire")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Dispositif à utiliser pour l'inférence")
    
    # Arguments de génération
    parser.add_argument("--prompts", nargs="+", type=str,
                        help="Prompts textuels pour générer des images")
    parser.add_argument("--prompt_file", type=str,
                        help="Fichier contenant des prompts (un par ligne)")
    parser.add_argument("--num_inference_steps", nargs="+", type=int, default=[1, 4],
                        help="Nombre d'étapes d'inférence à utiliser")
    parser.add_argument("--guidance_scale", type=float, default=7.5,
                        help="Échelle de guidage sans classification")
    parser.add_argument("--height", type=int, default=512,
                        help="Hauteur des images générées")
    parser.add_argument("--width", type=int, default=512,
                        help="Largeur des images générées")
    parser.add_argument("--output_dir", type=str, default="generated",
                        help="Répertoire pour sauvegarder les images générées")
    
    args = parser.parse_args()
    
    main(args) 
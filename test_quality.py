import os
import argparse
import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from model import PokemonADD
from tqdm import tqdm
import time

def main(args):
    """Generate and compare Pokemon images at different quality levels"""
    # Set device
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    print(f"Loading ADD model from {args.model_path}...")
    add_model = PokemonADD(
        pretrained_model_name_or_path=args.pretrained_model,
        enable_xformers=args.enable_xformers,
        device=device
    )
    
    # Load saved student weights
    if os.path.exists(args.model_path):
        checkpoint = torch.load(args.model_path, map_location=device)
        if 'student_state_dict' in checkpoint:
            add_model.student.load_state_dict(checkpoint['student_state_dict'])
            print("Successfully loaded student model weights")
        else:
            print("Could not find student weights in checkpoint, using pretrained weights")
    else:
        print(f"Warning: Model path {args.model_path} does not exist. Using pretrained weights.")
    
    # Test prompts
    test_prompts = [
        "a cute Pikachu with yellow fur and red cheeks",
        "a fierce Charizard with flaming tail",
        "a small Squirtle with blue shell",
        "a powerful Mewtwo with glowing eyes"
    ]
    
    # Set model to evaluation mode
    add_model.eval()
    
    # Generate images at different step counts
    all_step_counts = [1, 2, 4, 8, 16]
    step_times = {}
    
    # Create comparison grid
    plt.figure(figsize=(20, 15))
    
    for prompt_idx, prompt in enumerate(test_prompts):
        print(f"\nGenerating images for prompt: '{prompt}'")
        
        # Generate images for each step count
        images_for_prompt = {}
        
        for step_count_idx, step_count in enumerate(all_step_counts):
            print(f"  Generating with {step_count} step(s)...")
            
            # Time the generation
            start_time = time.time()
            
            # Generate image
            with torch.no_grad():
                images = add_model.generate(
                    [prompt],
                    num_inference_steps=step_count,
                    guidance_scale=args.guidance_scale,
                    height=args.height,
                    width=args.width
                )
            
            # Calculate time
            end_time = time.time()
            gen_time = end_time - start_time
            
            if step_count not in step_times:
                step_times[step_count] = []
            step_times[step_count].append(gen_time)
            
            # Save image
            image = images[0]
            images_for_prompt[step_count] = image
            
            # Save individual image
            img_pil = Image.fromarray(image)
            safe_prompt = "".join([c if c.isalnum() else "_" for c in prompt])[:30]
            filename = f"{safe_prompt}_{step_count}steps.png"
            img_path = os.path.join(args.output_dir, filename)
            img_pil.save(img_path)
            
            # Add to plot
            plt_idx = prompt_idx * len(all_step_counts) + step_count_idx + 1
            plt.subplot(len(test_prompts), len(all_step_counts), plt_idx)
            plt.imshow(image)
            plt.title(f"{step_count} step(s) - {gen_time:.2f}s")
            plt.axis('off')
        
    # Save comparison grid
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "step_comparison.png"), dpi=150)
    
    # Print timing statistics
    print("\n=== Generation Time Statistics ===")
    for step_count in all_step_counts:
        avg_time = sum(step_times[step_count]) / len(step_times[step_count])
        print(f"{step_count} step(s): {avg_time:.3f}s per image")
    
    print(f"\nComparison image saved to {os.path.join(args.output_dir, 'step_comparison.png')}")
    print(f"Individual images saved to {args.output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Pokemon image quality at different step counts")
    
    # Model arguments
    parser.add_argument("--model_path", type=str, default="output/models/add_model_final.pt",
                        help="Path to trained model checkpoint")
    parser.add_argument("--pretrained_model", type=str, default="CompVis/stable-diffusion-v1-4",
                        help="Pretrained model to initialize from")
    parser.add_argument("--enable_xformers", action="store_true",
                        help="Use xformers for memory-efficient attention")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use for inference")
    
    # Generation arguments
    parser.add_argument("--guidance_scale", type=float, default=9.0,
                        help="Classifier-free guidance scale")
    parser.add_argument("--height", type=int, default=512,
                        help="Height of generated images")
    parser.add_argument("--width", type=int, default=512,
                        help="Width of generated images")
    parser.add_argument("--output_dir", type=str, default="quality_test",
                        help="Directory to save generated images")
    
    args = parser.parse_args()
    
    main(args) 
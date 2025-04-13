import os
import argparse
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from tqdm import tqdm

from diffusers import DDIMScheduler

from modelADD import PokemonADD


# Dataset
class PokemonDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        import pandas as pd
        self.data = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = row['image_path']
        caption = row['caption']

        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return {'image': image, 'caption': caption}



# Loss functions
def hinge_d_loss(real_logits_list, fake_logits_list, r1_reg=0.0, real_imgs=None):
    # -- Hinge loss: real 
    d_loss_real = 0
    for logits in real_logits_list:
        d_loss_real += torch.mean(F.relu(1 - logits))
    d_loss_real /= len(real_logits_list)

    # -- Hinge loss: fake 
    d_loss_fake = 0
    for logits in fake_logits_list:
        d_loss_fake += torch.mean(F.relu(1 + logits))
    d_loss_fake /= len(fake_logits_list)

    d_loss = d_loss_real + d_loss_fake

    # -- R1 penalty (OPTION)
    if r1_reg > 0 and real_imgs is not None:
        real_imgs.requires_grad_(True)
        grad_sum = 0
        for logits in real_logits_list:
            grad = torch.autograd.grad(
                outputs=logits.sum(),
                inputs=real_imgs,
                create_graph=True,
                retain_graph=True,
                only_inputs=True,
                allow_unused=True
            )[0]
            if grad is not None:
                grad_sum += grad.pow(2).mean()

        if grad_sum > 0:
            grad_sum = grad_sum / len(real_logits_list)
            d_loss += r1_reg * 0.5 * grad_sum

        real_imgs.requires_grad_(False)

    return d_loss


def hinge_g_loss(fake_logits_list):
    g_loss = 0
    for logits in fake_logits_list:
        g_loss += -torch.mean(logits)
    g_loss /= len(fake_logits_list)
    return g_loss

# Train
def train_add(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)

    # -- Transform / Dataset
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    dataset = PokemonDataset(args.data_csv, transform=transform)
    dataloader = DataLoader(dataset, batch_size=args.batch_size,
                            shuffle=True, num_workers=4)

    # -- Build ADD model
    add_model = PokemonADD(
        pretrained_model_name_or_path=args.pretrained_model,
        device=device
    ).to(device)

    # -- Noise scheduler
    noise_scheduler = DDIMScheduler.from_pretrained(
        args.pretrained_model, subfolder="scheduler"
    )
    if hasattr(noise_scheduler, 'alphas_cumprod'):
        noise_scheduler.alphas_cumprod = noise_scheduler.alphas_cumprod.to(device)

    # -- Optimizers
    gen_params = list(add_model.student.parameters())
    disc_params = list(add_model.discriminator.heads.parameters())

    optimizer_g = optim.AdamW(gen_params, lr=args.lr_student, betas=(0.9, 0.999))
    optimizer_d = optim.AdamW(disc_params, lr=args.lr_disc, betas=(0.9, 0.999))

    # -- niveaux de bruits
    partial_levels = [float(x.strip()) for x in args.partial_noise_levels.split(",")]
    print(f"Partial noise levels used: {partial_levels}")

    global_step = 0
    for epoch in range(args.num_epochs):
        for batch in tqdm(dataloader, desc=f"Epoch {epoch}"):
            global_step += 1
            real_images = batch['image'].to(device)
            captions = batch['caption']

            # -- Encode text
            with torch.no_grad():
                text_embeds, _ = add_model.encode_text(captions)
                text_cond = text_embeds

            # -- bruit alétoire
            student_noise_level = np.random.choice(partial_levels)
            distill_noise_level = 1.0  # on peut aussi s'amuser à le paramétrer

          
            # Forward pass (Discriminator Update)
            out = add_model(
                real_images,
                text_cond,
                student_noise_level,
                noise_scheduler,
                distill_noise_level
            )
            disc_fake_logits = out["disc_fake_logits"]
            disc_real_logits = out["disc_real_logits"]

            optimizer_d.zero_grad()
            # Hinge loss Discriminateur (sans R1)
            d_loss = hinge_d_loss(
                disc_real_logits, disc_fake_logits,
                r1_reg=0.0  # R1 sera ajouté ci-dessous
            )

            # (opption : avec R1)
            if args.r1_reg > 0:
                real_images_r1 = real_images.detach().clone().requires_grad_(True)
                with torch.enable_grad():
                    # Embedding for real images
                    x0_embed = add_model.discriminator.feature_net(real_images_r1)
                    disc_text_cond2 = text_cond.mean(dim=1) if len(text_cond.shape) == 3 else text_cond
                    r1_real_logits = add_model.discriminator(real_images_r1, x0_embed, disc_text_cond2)
                    r1_penalty = 0
                    for logits in r1_real_logits:
                        grad = torch.autograd.grad(
                            outputs=logits.sum(),
                            inputs=real_images_r1,
                            create_graph=True,
                            retain_graph=True,
                            allow_unused=True
                        )[0]
                        if grad is not None:
                            r1_penalty += grad.pow(2).sum(dim=(1,2,3)).mean()
                    if r1_penalty > 0:
                        r1_penalty = r1_penalty / len(r1_real_logits)
                        d_loss_r1 = args.r1_reg * 0.5 * r1_penalty
                        d_loss = d_loss + d_loss_r1

                real_images_r1.requires_grad_(False)
                del real_images_r1

            d_loss.backward()
            optimizer_d.step()

            # On refait un forward pass
            out_g = add_model(
                real_images,
                text_cond,
                student_noise_level,
                noise_scheduler,
                distill_noise_level
            )
            student_latents_g = out_g["student_out"]
            teacher_latents_g = out_g["teacher_out"]
            disc_fake_logits_g = out_g["disc_fake_logits"]

            optimizer_g.zero_grad()

            # -- Distillation
            t_int_s = int(student_noise_level * 999)
            alpha_s = noise_scheduler.alphas_cumprod[t_int_s].sqrt().to(device)
            distill_weight = alpha_s  # ou 1.0, ou une autre fonction c(t)

            if args.use_pixel_dist:
                # Distillation en pixel space
                with torch.no_grad():
                    teacher_img = add_model.decode_latents_to_images(teacher_latents_g)
                student_img = add_model.decode_latents_to_images(student_latents_g)
                distillation_loss = F.mse_loss(student_img, teacher_img) * distill_weight
            else:
                # Distillation en latent space (défaut)
                distillation_loss = F.mse_loss(student_latents_g, teacher_latents_g) * distill_weight

            # -- Adversarial
            g_loss_adv = hinge_g_loss(disc_fake_logits_g)

            g_loss = distillation_loss + args.lambda_adv * g_loss_adv
            g_loss.backward()
            optimizer_g.step()

            if global_step % 50 == 0:
                print(f"Step {global_step}"
                      f" | d_loss {d_loss.item():.4f}"
                      f" | g_loss {g_loss.item():.4f}"
                      f" | distill {distillation_loss.item():.4f}"
                      f" | adv {g_loss_adv.item():.4f}")

        # Save checkpoint
        '''
        ckpt_path = os.path.join(args.output_dir, f"checkpoint_{epoch+1}.pth")
        torch.save({
            "student": add_model.student.state_dict(),
            "disc_heads": add_model.discriminator.heads.state_dict(),
            "optimizer_g": optimizer_g.state_dict(),
            "optimizer_d": optimizer_d.state_dict(),
            "epoch": epoch
        }, ckpt_path)
        print(f"Saved checkpoint to {ckpt_path}")'''
        
        # À la fin de la dernière epoch seulement
        if epoch == args.num_epochs - 1:
            ckpt_path = os.path.join(args.output_dir, "final_checkpoint.pth")
            torch.save({
                "student": add_model.student.state_dict(),
                "disc_heads": add_model.discriminator.heads.state_dict(),
                "optimizer_g": optimizer_g.state_dict(),
                "optimizer_d": optimizer_d.state_dict(),
                "epoch": epoch
            }, ckpt_path)
            print(f"Final model saved at: {ckpt_path}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_csv", type=str, default="data/pokemon_train.csv")
    parser.add_argument("--output_dir", type=str, default="output_add")
    parser.add_argument("--pretrained_model", type=str, default="CompVis/stable-diffusion-v1-4")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_epochs", type=int, default=5)

    # Hyperparams: Student / Discriminator learning rates
    parser.add_argument("--lr_student", type=float, default=5e-6, help="LR for generator/Student")
    parser.add_argument("--lr_disc", type=float, default=5e-6, help="LR for discriminator heads")

    # Adversarial weighting
    parser.add_argument("--lambda_adv", type=float, default=1.5, help="Weight of adversarial term in G loss")

    # R1 penalty
    parser.add_argument("--r1_reg", type=float, default=1e-6, help="R1 penalty weight (0 to disable)")

    # Partial noise levels
    parser.add_argument("--partial_noise_levels", type=str, default="0.0,0.25,0.5,1.0",
                        help="Comma-separated list of partial noise levels to sample from")

    # Distillation dans l'espace pixel ?
    parser.add_argument("--use_pixel_dist", action="store_true",
                        help="If set, do the MSE distillation in pixel space instead of latent space")

    args = parser.parse_args()

    train_add(args)

modelADD.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import UNet2DConditionModel, AutoencoderKL, DDIMScheduler
from transformers import CLIPTextModel, CLIPTokenizer
import numpy as np
from tqdm import tqdm
import timm  # pour charger le ViT pré-entraîné 

class FrozenViTFeatureExtractor(nn.Module):
    """
    Extrait plusieurs features du ViT (pas seulement la sortie finale) pour améliorer la capacité du discriminateur.
    """
    def __init__(self, vit_name="vit_small_patch16_224_dino"):
        super().__init__()
        self.vit = timm.create_model(vit_name, pretrained=True, features_only=True,
                                     # features_only=True permet de récupérer
                                     # plusieurs features intermédiaires
                                     out_indices=(2, 3, 4))  
        
        for param in self.vit.parameters():
            param.requires_grad = False
        self.vit.eval()

        self.img_size = 224

        # out_channels=[384, 384, 384] => total 384*3 = 1152
        self.out_channels = sum(self.vit.feature_info.channels())

    def forward(self, x):
        """ x: (B, 3, H, W) -> Retourne la concaténation des features sur plusieurs blocs."""

        x_in = x.clone()
        
        # Resize si besoin 
        _, _, H, W = x_in.shape
        if H != self.img_size or W != self.img_size:
            x_in = F.interpolate(x_in, size=(self.img_size, self.img_size),
                              mode='bicubic', align_corners=False)

        with torch.no_grad():
            feats = self.vit(x_in)  # list de Tensors, ex. [(B,384,14,14), (B,384,7,7), ...]
        
        # On va global-pool chaque feature map en un (B, c) avec adaptive avgpool
        pooled_list = []
        for f in feats:
            # f.shape = (B, C, Hf, Wf)
            pooled_f = F.adaptive_avg_pool2d(f, (1, 1))  # => (B, C, 1, 1)
            pooled_f = pooled_f.squeeze(-1).squeeze(-1) # => (B, C)
            pooled_list.append(pooled_f)
        # Concat
        emb = torch.cat(pooled_list, dim=1)  # (B, sum_of_channels)
        return emb


class ADDDiscriminator(nn.Module):
    """
    Discriminateur:
      - Récupère plusieurs features du ViT gelé (FrozenViTFeatureExtractor).
      - Applique quelques têtes MLP sur la concat (feat + x0 + text_emb).
    """
    def __init__(self, feat_dim=1152, cond_dim=768, n_heads=3):
        super().__init__()
        self.feature_net = FrozenViTFeatureExtractor("vit_small_patch16_224_dino")

        hidden = 256
        heads = []
        for _ in range(n_heads):
            head = nn.Sequential(
                nn.Linear(feat_dim + feat_dim + cond_dim, hidden),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(hidden, 1)
            )
            heads.append(head)
        self.heads = nn.ModuleList(heads)

    def forward(self, x_fake_or_real, x0_embed, text_embed):
        """
        x_fake_or_real: (B, 3, H, W)
        x0_embed: (B, feat_dim) embedding real image
        text_embed: (B, cond_dim)
        """
        # Extraction des features
        feat = self.feature_net(x_fake_or_real)  # (B, feat_dim=1152)
        
        if x0_embed.shape[1] != feat.shape[1]:
            # Resize x0_embed to match feat if needed
            x0_embed = x0_embed.expand(feat.shape[0], feat.shape[1])
            
        if text_embed.dim() > 2:
            text_embed = text_embed.mean(dim=1)
        
        combined = torch.cat([feat, x0_embed, text_embed], dim=1)
        
        # Application des têtes
        logits_list = []
        for head in self.heads:
            logits = head(combined)
            logits_list.append(logits)
        return logits_list


class PokemonADD(nn.Module):
    def __init__(
        self,
        pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4",
        device="cuda" if torch.cuda.is_available() else "cpu"
    ):
        super().__init__()
        self.device = device

        # 1) Text
        self.tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_name_or_path, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(pretrained_model_name_or_path, subfolder="text_encoder").to(device)
        for param in self.text_encoder.parameters():
            param.requires_grad = False

        # 2) VAE
        self.vae = AutoencoderKL.from_pretrained(pretrained_model_name_or_path, subfolder="vae").to(device)
        for param in self.vae.parameters():
            param.requires_grad = False

        # 3) Student U-Net (entraîné)
        self.student = UNet2DConditionModel.from_pretrained(
            pretrained_model_name_or_path, subfolder="unet"
        ).to(device)
        # 4) Teacher U-Net (figé)
        self.teacher = UNet2DConditionModel.from_pretrained(
            pretrained_model_name_or_path, subfolder="unet"
        ).to(device)
        for param in self.teacher.parameters():
            param.requires_grad = False

        # 5) Discriminateur
        self.discriminator = ADDDiscriminator(
            feat_dim=1152,  # somme des channels du ViT multi-couches
            cond_dim=768,   # clip hidden size
            n_heads=3
        ).to(device)

    # Encodage texte
    def encode_text(self, prompts):
        text_input = self.tokenizer(
            prompts,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)
        with torch.no_grad():
            text_outputs = self.text_encoder(text_input.input_ids)
            text_embeddings = text_outputs.last_hidden_state
            pooled = text_outputs.pooler_output if text_outputs.pooler_output is not None else text_embeddings[:, 0, :]
        return text_embeddings, pooled

    # Encode image -> latents
    def encode_images_to_latents(self, images):
        if images.min() < 0:
            images = (images + 1) / 2.0
        with torch.no_grad():
            latents = self.vae.encode(images).latent_dist.sample()
            latents = latents * 0.18215
        return latents

    # Decode latents -> image
    def decode_latents_to_images(self, latents):
        with torch.no_grad():
            latents = latents / 0.18215
            images = self.vae.decode(latents).sample
            images = (images / 2 + 0.5).clamp(0,1)
        return images

    def re_diffuse_student_output_for_teacher(
        self,
        student_out,
        teacher_noise_level,
        noise_scheduler,
        sequence_embeddings
    ):
        t_int = int(teacher_noise_level * 999)
        noise = torch.randn_like(student_out)
        alpha_t = noise_scheduler.alphas_cumprod[t_int].sqrt().to(self.device)
        sigma_t = (1 - noise_scheduler.alphas_cumprod[t_int]).sqrt().to(self.device)

        noised_latents = alpha_t * student_out + sigma_t * noise

        t_tensor = torch.tensor([t_int]*student_out.shape[0], dtype=torch.long, device=self.device)
        with torch.no_grad():
            teacher_pred = self.teacher(
                noised_latents,
                t_tensor,
                encoder_hidden_states=sequence_embeddings
            ).sample
        return teacher_pred

    def forward(
        self,
        real_images,
        text_embeddings,
        partial_noise_level,
        noise_scheduler,
        distill_noise_level=1.0
    ):
        B = real_images.size(0)

        # Vérif: text_embeddings = (B, 77, 768)
        if len(text_embeddings.shape) != 3:
            raise ValueError("Text embeddings must be (B, seq_len, hidden_dim).")

        # 1) encode en latents
        latents_0 = self.encode_images_to_latents(real_images)

        # 2) partial diffuse
        t_int = int(partial_noise_level * 999)
        alpha_s = noise_scheduler.alphas_cumprod[t_int].sqrt().to(self.device)
        sigma_s = (1 - noise_scheduler.alphas_cumprod[t_int]).sqrt().to(self.device)
        noise = torch.randn_like(latents_0)
        latents_s = alpha_s * latents_0 + sigma_s * noise

        # 3) Student
        t_tensor = torch.tensor([t_int]*B, dtype=torch.long, device=self.device)
        student_out = self.student(
            latents_s,
            t_tensor,
            encoder_hidden_states=text_embeddings
        ).sample  # (B,4,64,64) => prédiction d'epsilon si c'est stable diffusion

        # 4) Re-noise => Teacher
        teacher_pred = self.re_diffuse_student_output_for_teacher(
            student_out,
            distill_noise_level,
            noise_scheduler,
            text_embeddings
        )

        # 5) Decode student_out pour l'adversarial
        student_img_out = self.decode_latents_to_images(student_out)

        # Calculer x0_embed (embedding image réelle) - only once, with no_grad for stability
        with torch.no_grad():
            x0_embed = self.discriminator.feature_net(real_images)

        disc_text_cond = text_embeddings.mean(dim=1)  # (B,768)

        # Logits réels - passez x0_embed pré-calculé pour l'efficacité et la stabilité du gradient
        real_logits_list = self.discriminator(real_images, x0_embed, disc_text_cond)
        
        # # Faux logits - transmettez x0_embed pré-calculé pour plus de cohérence avec la façon dont les vrais logits sont calculés
        fake_logits_list = self.discriminator(student_img_out, x0_embed, disc_text_cond)

        return {
            "student_out": student_out,
            "teacher_out": teacher_pred,
            "student_img_out": student_img_out,
            "real_img": real_images,
            "disc_fake_logits": fake_logits_list,
            "disc_real_logits": real_logits_list,
            "latents_s": latents_s,
            "latents_0": latents_0
        }
    
    def generate(self, prompts, num_inference_steps=40, guidance_scale=7.5, height=512, width=512, seed=None):
        if isinstance(prompts, str):
            prompts = [prompts]
            
        if seed is not None:
            torch.manual_seed(seed)
        
        self.eval()
        batch_size = len(prompts)
        device = self.device
        
        # Initialiser le planificateur DDIM
        scheduler = DDIMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=False
        )
        scheduler.set_timesteps(num_inference_steps)
        
        # Encoder les prompts texte
        text_embeddings, _ = self.encode_text(prompts)  # (B, 77, 768)
        
        # Générer des embeddings non conditionnels pour la guidance
        null_prompts = [""] * batch_size
        null_embeddings, _ = self.encode_text(null_prompts)
        combined_embeddings = torch.cat([null_embeddings, text_embeddings], dim=0)  # (2B, 77, 768)
        
        # Initialiser les latents
        latents = torch.randn(
            (batch_size, 4, height // 8, width // 8),
            device=device,
            dtype=torch.float32
        )
        
        # Boucle de débruitage
        with torch.no_grad():
            for t in tqdm(scheduler.timesteps, desc="Débruitage"):
                # Dupliquer les latents pour conditionnel et non conditionnel
                latent_input = torch.cat([latents] * 2, dim=0)  # (2B, 4, H//8, W//8)
                
                # Préparer l'étape temporelle
                timestep = torch.full((latent_input.shape[0],), t, device=device, dtype=torch.long)
                
                # Prédiction du modèle étudiant
                noise_pred = self.student(
                    latent_input,
                    timestep,
                    encoder_hidden_states=combined_embeddings
                ).sample
                
                # Guidance sans classifieur
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                
                # Mettre à jour les latents avec le planificateur
                latents = scheduler.step(noise_pred, t, latents).prev_sample
        
        # Décoder les latents en images
        images = self.decode_latents_to_images(latents)
        
        # Convertir en tableaux numpy
        images_np = [np.uint8(img.permute(1, 2, 0).cpu().numpy() * 255) for img in images]
        
        return images_np
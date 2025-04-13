import streamlit as st
import os
import torch
from diffusers import StableDiffusionPipeline, AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer

st.title("Génération d'image avec Stable Diffusion - Trois Modèles")

# Chemins des modèles
model_name = "CompVis/stable-diffusion-v1-4"  # modèle de base
finetuned_model_path = "output_teacher/model12k"  # pipeline fine-tunée
adv_checkpoint_path = "output_add/model3/checkpoint_5.pth"  # checkpoint adversarial distillation

# Déterminer le device
device = "cuda" if torch.cuda.is_available() else "cpu"

# ----------------------------------------
# Fonctions de chargement de pipeline
# ----------------------------------------

@st.cache_resource(show_spinner=False)
def load_pipeline_baseline():
    """Charge la pipeline basique depuis le modèle de base."""
    pipe = StableDiffusionPipeline.from_pretrained(
        model_name,
        torch_dtype=torch.float16
    ).to(device)
    return pipe

@st.cache_resource(show_spinner=False)
def load_pipeline_finetuned():
    """Charge la pipeline fine-tunée depuis le dossier sauvegardé."""
    pipe = StableDiffusionPipeline.from_pretrained(
        finetuned_model_path,
        torch_dtype=torch.float16
    ).to(device)
    return pipe

@st.cache_resource(show_spinner=False)
def load_pipeline_adversarial():
    """
    Charge la pipeline de base, puis remplace les poids du UNet par ceux issus 
    de la distillation adversariale à partir du checkpoint.
    """
    # Charger les composants du modèle de base
    tokenizer = CLIPTokenizer.from_pretrained(model_name, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(model_name, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(model_name, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(model_name, subfolder="unet")
    scheduler = DDPMScheduler.from_pretrained(model_name, subfolder="scheduler")
    
    # Charger les poids adversariaux pour le UNet
    checkpoint = torch.load(adv_checkpoint_path, map_location=device)
    unet.load_state_dict(checkpoint, strict=False)
    
    # Placer les modèles sur le device
    text_encoder.to(device)
    unet.to(device)
    vae.to(device)
    
    # Créer la pipeline en réassemblant les composants
    pipe = StableDiffusionPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        scheduler=scheduler,
        safety_checker=None  # Activez le safety_checker si nécessaire
    )
    pipe.to(device)
    return pipe

# Sélection du modèle à utiliser
model_choice = st.selectbox(
    "Choisissez le modèle à utiliser :",
    ["Basique", "Fine-Tuned", "Adversarial Distillation"]
)

if model_choice == "Basique":
    pipe = load_pipeline_baseline()
elif model_choice == "Fine-Tuned":
    pipe = load_pipeline_finetuned()
else:
    pipe = load_pipeline_adversarial()

# ----------------------------------------
# Interface utilisateur pour les paramètres
# ----------------------------------------

st.header("Paramètres de génération")
prompt = st.text_input("Entrez votre prompt", value="A beautiful fantasy landscape")
num_inference_steps = st.slider("Nombre d'étapes d'inférence", min_value=10, max_value=100, value=50, step=1)
guidance_scale = st.slider("Guidance Scale", min_value=1.0, max_value=10.0, value=7.5, step=0.1)

# ----------------------------------------
# Bouton de génération et affichage du résultat
# ----------------------------------------

if st.button("Générer l'image"):
    st.write("Génération en cours...")
    with st.spinner("Traitement en cours..."):
        # Utiliser torch.autocast pour optimiser l'inférence sur GPU
        with torch.autocast(device):
            output = pipe(
                prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale
            )
        image = output.images[0]
    st.image(image, caption=f"Image générée avec le modèle {model_choice}", use_column_width=True)

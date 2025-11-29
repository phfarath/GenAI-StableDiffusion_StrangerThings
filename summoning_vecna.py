import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
from diffusers import UNet2DConditionModel, AutoencoderKL, LMSDiscreteScheduler
from transformers import CLIPTextModel, CLIPTokenizer

# --- CONFIGURAÇÃO DO RITUAL (Hyperparameters) ---
DEVICE = "cuda" # Para GPU da NVIDIA ("mps" para Mac M1/M2/M3 ou "cpu" caso tenha apenas CPU, isso irá demorar mais tempo)
GUIDANCE_SCALE = 7.5 # Força Psíquica da Eleven
STEPS = 50 # Tempo de materialização
SEED = 1983 # Ano do incidente em Hawkins

def open_gate():
    print("⏳ Carregando modelos do Mundo Invertido...")
    # 1. O PORTAL (VAE): Traduz pixels reais para o Espaço Latente
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(DEVICE)
    
    # 2. O CÉREBRO (UNet): A rede neural que entende como remover o ruído
    unet = UNet2DConditionModel.from_pretrained("Lykon/DreamShaper", subfolder="unet").to(DEVICE)
    
    # 3. O TRADUTOR (Tokenizer/Encoder): Converte palavras em embeddings matemáticos
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").to(DEVICE)
    
    # 4. O TEMPO (Scheduler): Define como o ruído se transforma em imagem
    scheduler = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear")
    
    return vae, unet, tokenizer, text_encoder, scheduler

def summon_entity(prompt):
    vae, unet, tokenizer, text_encoder, scheduler = open_gate()
    
    # PASSO 1: ENTRANDO NO VAZIO (Gerando Latents)
    # Começamos com pura estática/ruído gaussiano. O "Mind Flayer" em forma de partículas.
    generator = torch.manual_seed(SEED)
    latents = torch.randn((1, 4, 64, 64), generator=generator)
    latents = latents.to(DEVICE)
    latents = latents * scheduler.init_noise_sigma

    # PASSO 2: FOCALIZAÇÃO (Text Conditioning)
    # "Eleven, eu preciso que você encontre..." 
    text_input = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
    text_embeddings = text_encoder(text_input.input_ids.to(DEVICE))[0]

    max_length = text_input.input_ids.shape[-1]
    uncond_input = tokenizer([""], padding="max_length", max_length=max_length, return_tensors="pt")
    uncond_embeddings = text_encoder(uncond_input.input_ids.to(DEVICE))[0]
    text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

    # PASSO 3: MATERIALIZAÇÃO (Denoising Loop)
    scheduler.set_timesteps(STEPS)
    
    print(f"⚡ Invocando: '{prompt}' do Mundo Invertido...")
    
    for t in scheduler.timesteps:
        # Expandir latents para guiar a geração (Classifier Free Guidance)
        latent_model_input = torch.cat([latents] * 2)
        latent_model_input = scheduler.scale_model_input(latent_model_input, t)

        # A UNet prevê onde está o "monstro" no meio do ruído
        with torch.no_grad():
            noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

        # Aplicando a "Força Psíquica" (Guidance Scale)
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + GUIDANCE_SCALE * (noise_pred_text - noise_pred_uncond)

        # Removendo o ruído: O monstro ganha forma sólida
        latents = scheduler.step(noise_pred, t, latents).prev_sample

    # PASSO 4: ABRINDO A FENDA (Decoding)
    # Trazendo a imagem do espaço latente (matemático) para o pixel space (visível)
    latents = 1 / 0.18215 * latents
    with torch.no_grad():
        image = vae.decode(latents).sample

    # Processamento final da imagem
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).numpy()
    image = (image * 255).round().astype("uint8")

    # Criamos a variável pil_image explicitamente
    pil_image = Image.fromarray(image[0])
    
    # Salvamos no disco
    output_filename = "vecna_summoned.png"
    pil_image.save(output_filename)
    
    # Tentamos abrir a imagem (se falhar, o script não quebra)
    try:
        pil_image.show()
    except:
        pass

    return output_filename

# --- EXECUÇÃO ---
if __name__ == "__main__":
    result = summon_entity("A terrifying creature made of vines and red lightning, cinematic lighting, 8k") # "prompt" - o que você quer que seja gerado
    # --- EXECUÇÃO ---
if __name__ == "__main__":
    try:
        filename = summon_entity("A terrifying creature made of vines and red lightning, cinematic lighting, 8k, stranger things style")
        print(f"\n🩸 Sangramento nasal detectado! A imagem foi salva como '{filename}' na pasta do projeto.")
    except Exception as e:
        print(f"\n❌ ERRO CRÍTICO: {e}")

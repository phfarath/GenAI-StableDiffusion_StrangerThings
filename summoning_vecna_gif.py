import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
from diffusers import UNet2DConditionModel, AutoencoderKL, LMSDiscreteScheduler
from transformers import CLIPTextModel, CLIPTokenizer

# --- CONFIGURAÇÃO ---
# Hardware
if torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu" # Vai demorar, mas funciona

# Parâmetros
GUIDANCE_SCALE = 7.5 
STEPS = 30 
SEED = 1983 
OUTPUT_FOLDER = "frames_vecna"
PROMPT = "A terrifying creature made of vines and red lightning, cinematic lighting, 8k, stranger things style"

# Cria a pasta de frames se não existir
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def open_gate():
    print("⏳ Carregando modelos...")
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(DEVICE)
    unet = UNet2DConditionModel.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="unet").to(DEVICE)
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").to(DEVICE)
    scheduler = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear")
    return vae, unet, tokenizer, text_encoder, scheduler

def decode_and_save_frame(latents, vae, step_num):
    """Função auxiliar para decodificar e salvar o frame intermediário"""
    with torch.no_grad():
        # Escalar latents de volta para o VAE
        latents_decoded = 1 / 0.18215 * latents
        image = vae.decode(latents_decoded).sample

    # Converter para imagem
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).numpy()
    image = (image * 255).round().astype("uint8")
    pil_image = Image.fromarray(image[0])

    # Adicionar número do passo na imagem (Opcional, estilo "rec")
    draw = ImageDraw.Draw(pil_image)
    # Tenta usar fonte padrão ou arial, senão usa a default
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        font = ImageFont.load_default()
        
    draw.text((10, 10), f"DIFFUSION STEP: {step_num}/{STEPS}", fill=(255, 0, 0), font=font)
    
    # Salvar
    filename = f"{OUTPUT_FOLDER}/frame_{step_num:03d}.png"
    pil_image.save(filename)
    return pil_image

def create_animation():
    vae, unet, tokenizer, text_encoder, scheduler = open_gate()
    
    # 1. Latents Iniciais (O Ruído Puro)
    generator = torch.manual_seed(SEED)
    latents = torch.randn((1, 4, 64, 64), generator=generator)
    latents = latents.to(DEVICE)
    latents = latents * scheduler.init_noise_sigma

    # 2. Prompt Embeddings
    text_input = tokenizer(PROMPT, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
    text_embeddings = text_encoder(text_input.input_ids.to(DEVICE))[0]
    
    max_length = text_input.input_ids.shape[-1]
    uncond_input = tokenizer([""], padding="max_length", max_length=max_length, return_tensors="pt")
    uncond_embeddings = text_encoder(uncond_input.input_ids.to(DEVICE))[0]
    text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

    # 3. Loop de Difusão
    scheduler.set_timesteps(STEPS)
    print(f"⚡ Gerando animação frame-a-frame...")
    
    frames = []

    # Salvamos o frame 0 (ruído puro)
    print("📸 Capturando ruído inicial...")
    frames.append(decode_and_save_frame(latents, vae, 0))

    for i, t in enumerate(scheduler.timesteps):
        # Expandir latents
        latent_model_input = torch.cat([latents] * 2)
        latent_model_input = scheduler.scale_model_input(latent_model_input, t)

        # Predizer ruído
        with torch.no_grad():
            noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

        # Guidance
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + GUIDANCE_SCALE * (noise_pred_text - noise_pred_uncond)

        # Step do scheduler
        latents = scheduler.step(noise_pred, t, latents).prev_sample
        
        # --- A MÁGICA ACONTECE AQUI ---
        # A cada passo (ou a cada 2 passos para ser mais rápido), salvamos uma imagem
        # Decodificar o VAE a cada frame é pesado, mas necessário para ver a imagem
        print(f"📸 Capturando frame {i+1}/{STEPS}...")
        frame_img = decode_and_save_frame(latents, vae, i+1)
        frames.append(frame_img)

    # 4. Criar GIF
    print("🎬 Montando GIF animado...")
    # O frame final fica mais tempo na tela (duration maior)
    frames[0].save(
        "vecna_transformation.gif",
        save_all=True,
        append_images=frames[1:],
        duration=150, # 150ms por frame
        loop=0
    )
    print(f"✅ GIF salvo com sucesso: vecna_transformation.gif")

if __name__ == "__main__":
    create_animation()

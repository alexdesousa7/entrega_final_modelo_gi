from diffusers import StableDiffusionPipeline
import torch

# Prompt para comparar ambos modelos
prompt = "Medieval knight reading an ancient manuscript inside a stone chamber"

# 1. Imagen ANTES del finetuning

pipe_base = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    torch_dtype=torch.float32
)

image_before = pipe_base(prompt).images[0]
image_before.save("before.png")
print("Imagen 'before.png' generada con el modelo base.")

# 2. Imagen DESPUÉS del finetuning

pipe_ft = StableDiffusionPipeline.from_pretrained(
    "alexdesousa/sd-oldbook-finetuned-512-200",
    torch_dtype=torch.float32
)

image_after = pipe_ft(prompt).images[0]
image_after.save("after.png")
print("Imagen 'after.png' generada con el modelo finetuneado.")
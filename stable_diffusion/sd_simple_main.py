import torch
from diffusers import StableDiffusionPipeline

# pip install diffusers

model_ckpt = "CompVis/stable-diffusion-v1-4"
device = "mps" # cuda, cpu, mps
weight_dtype = torch.float16

pipe = StableDiffusionPipeline.from_pretrained(model_ckpt, torch_dtype=weight_dtype)
pipe = pipe.to(device)

prompt = "a photograph of an astronaut riding a horse"
image = pipe(prompt).images[0]
image.save("simple_results.png")
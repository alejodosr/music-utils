import random
import torch
from diffusers import LDMTextToImagePipeline

pipe = LDMTextToImagePipeline.from_pretrained("/home/alejandro/py_workspace/music-utils/checkpoints/stable-diffusion-v1-3-diffusers")

prompt = "photo of donald trump bleeding on the floor, screaming, photorealistic, high detail"
rand_num = random.randint(0, 1e4)
print(f'Random number: {rand_num}')
seed = torch.manual_seed(rand_num)
images = pipe([prompt], num_inference_steps=50, guidance_scale=7.5, generator=seed, batch_size=6)["sample"]

for img in images:
    img.show()




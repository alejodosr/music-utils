import random

from deepfloyd_if.pipelines import style_transfer

import requests

image_url = 'https://img.freepik.com/free-psd/juice-text-style-effect_136295-695.jpg'
image_path = '/tmp/image_name.jpg'

img_data = requests.get(image_url).content
with open(image_path, 'wb') as handler:
    handler.write(img_data)

image_path = '/tmp/estas_bien.png'

from PIL import Image
raw_pil_image = Image.open(image_path).convert("RGB")
raw_pil_image = raw_pil_image.resize((768, 512))

from deepfloyd_if.modules import IFStageI, IFStageII, StableStageIII
from deepfloyd_if.modules.t5 import T5Embedder

device = 'cuda:0'
if_I = IFStageI('IF-I-XL-v1.0', device=device)
if_II = IFStageII('IF-II-L-v1.0', device=device)
if_III = StableStageIII('stable-diffusion-x4-upscaler', device=device)
t5 = T5Embedder(device="cpu")

seed = random.randint(0, 2**32 - 1)
print("Seed: ", seed)

result = style_transfer(
    t5=t5, if_I=if_I, if_II=if_II,
    support_pil_img=raw_pil_image,
    style_prompt=[
        'in style of foliage, leaves, flowers, branches',
        'in style of construction gray bricks, concrete, cranes',
        'in style of water, 3d art',
        # 'in style of classic anime from 1990',
    ],
    seed=seed,
    if_I_kwargs={
        "guidance_scale": 10.0,
        "sample_timestep_respacing": "10,10,10,10,10,10,10,10,0,0",
        'support_noise_less_qsample_steps': 5,
    },
    if_II_kwargs={
        "guidance_scale": 7.0,
        "sample_timestep_respacing": 'smart50',
        "support_noise_less_qsample_steps": 5,
    },
)
if_I.show(result['II'], 1, 20)
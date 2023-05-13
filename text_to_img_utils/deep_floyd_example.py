import os
import random

from deepfloyd_if.modules import IFStageI, IFStageII, StableStageIII
from deepfloyd_if.modules.t5 import T5Embedder

device = 'cuda:0'
if_I = IFStageI('IF-I-XL-v1.0', device=device)
if_II = IFStageII('IF-II-L-v1.0', device=device)
if_III = StableStageIII('stable-diffusion-x4-upscaler', device=device)
t5 = T5Embedder(device="cpu")

from deepfloyd_if.pipelines import dream

prompt = 'text saying "?" made completely out of foliage, twigs, leaves and flowers, side view'
# prompt = 'text saying "?" made completely out of robotics parts and pieces, 3d pixel retro render, realistic'
count = 4

seed = random.randint(0, 2**32 - 1)
print('seed: ', seed)

result = dream(
    t5=t5, if_I=if_I, if_II=if_II, if_III=if_III,
    prompt=[prompt]*count,
    seed=seed,
    if_I_kwargs={
        "guidance_scale": 7.0,
        "sample_timestep_respacing": "smart100",
    },
    if_II_kwargs={
        "guidance_scale": 4.0,
        "sample_timestep_respacing": "smart50",
    },
    if_III_kwargs={
        "guidance_scale": 9.0,
        "noise_level": 20,
        "sample_timestep_respacing": "75",
    },
)

if_III.show(result['III'], size=14)
base_folder = '/home/alejandro/fresssh/releases/20221221_woman/video_carlos'

for i, img in enumerate(result['III']):
    img.save(os.path.join(base_folder, f'img_{seed}_{i}_{prompt}.png'))


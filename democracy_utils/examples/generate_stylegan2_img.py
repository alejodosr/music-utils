from io import BytesIO
import torch
from PIL import Image

in_img_path = '/home/alejandro/py_workspace/Caricature-Your-Face/data/content/ale_test.jpg'
out_img_path = '/tmp/ale_anime.png'

device = "cuda" if torch.cuda.is_available() else "cpu"
model = torch.hub.load("bryandlee/animegan2-pytorch:main", "generator", device=device).eval()
face2paint = torch.hub.load("bryandlee/animegan2-pytorch:main", "face2paint", device=device)
image_format = "png" #@param ["jpeg", "png"]

im_in = Image.open(in_img_path).convert("RGB")
im_out = face2paint(model, im_in, side_by_side=False)

im_out.save(out_img_path, format=image_format)

import os
from PIL import Image


in_folder = '/home/alejandro/fresssh/releases/20221221_woman/video_carlos/no_backkground/scaled'
out_folder = '/home/alejandro/fresssh/releases/20221221_woman/video_carlos/no_backkground/scaled'

os.system(f'mkdir {out_folder}')

# giving file extension
ext = ('.png', '.jpg', '.jpeg')

# iterating over all files
for file in os.listdir(in_folder):
    if file.endswith(ext):
        img = Image.open(os.path.join(in_folder, file))
        img.save(os.path.join(out_folder, file.replace('png', 'jpg')))
    else:
        continue
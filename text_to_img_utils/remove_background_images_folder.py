import os

in_folder = '/home/alejandro/fresssh/releases/20221221_woman/video_carlos/selected'
out_folder = '/home/alejandro/fresssh/releases/20221221_woman/video_carlos/selected/no_back'

os.system(f'mkdir {out_folder}')

# giving file extension
ext = ('.png', '.jpg', '.jpeg')

# iterating over all files
for file in os.listdir(in_folder):
    if file.endswith(ext):
        os.system(f"rembg i '{os.path.join(in_folder, file)}' '{os.path.join(out_folder, file)}'")
    else:
        continue


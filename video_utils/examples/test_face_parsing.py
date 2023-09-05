import os
import cv2
import time
import numpy as np
import torch
from argparse import ArgumentParser
from ibug.face_detection import RetinaFacePredictor
from ibug.face_parsing import FaceParser as RTNetPredictor
from ibug.face_parsing.utils import label_colormap

colormap = label_colormap(14)

face_detector = RetinaFacePredictor(threshold=0.8, device='cuda:0',
                                    model=(RetinaFacePredictor.get_model('mobilenet0.25')))
face_parser = RTNetPredictor(
    device='cuda:0', ckpt=None, encoder='rtnet50', decoder='fcn', num_classes=14)

img = cv2.imread('/home/alejandro/temp/test_image_ale.jpg')

start_time = time.time()
faces = face_detector(img, rgb=False)
alphas = np.linspace(0.75, 0.25, num=50)

if len(faces) == 0:
    exit()

masks = face_parser.predict_img(img, faces, rgb=False)
elapsed_time = time.time() - start_time

# Textural output
print(f'Frame processed in {elapsed_time * 1000.0:.04f} ms: ' +
      f'{len(faces)} faces detected.')

# # Rendering
dst = img
for i, (face, mask) in enumerate(zip(faces, masks)):
    bbox = face[:4].astype(int)
    cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color=(
        0, 0, 255), thickness=2)
    alpha = alphas[i]
    index = (mask > 0) & (
            (mask == 1) |  # Skin
            (mask == 2) |  # Left eyebrow
            (mask == 3) |  # Right eyebrow
            (mask == 4) |  # Left eye
            (mask == 5) |  # Right eye
            (mask == 6) |  # Nose
            (mask == 7) |  # Upper lip
            (mask == 8) |  # Inner Mouth
            (mask == 9) |  # Lower lip
            (mask == 11) |  # Left ear
            (mask == 12)  # Right ear
            )
    res = colormap[mask]
    dst[index] = (1 - alpha) * img[index].astype(float) + \
                 alpha * res[index].astype(float)
dst = np.clip(dst.round(), 0, 255).astype(np.uint8)
img = dst

cv2.imwrite("/home/alejandro/fresssh/releases/accumen/test_face_parsing.jpg", img)
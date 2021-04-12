import cv2
import os

RESOLUTION=(480, 480)
INPUT_DATASET = '/media/alejandro/TOSHIBA EXT/ale/ai_kitties_40000_512px'
OUTPUT_DATASET = '/media/alejandro/TOSHIBA EXT/ale/ai_kitties_40000_480px'

os.system('mkdir -p ' + '"' + OUTPUT_DATASET + '"')

for subdir, dirs, files in os.walk(INPUT_DATASET):
    for filename in files:
        filepath = subdir + os.sep + filename
        if filepath.endswith(".jpg") or filepath.endswith(".png"):
            try:
                print(filepath)
                img = cv2.imread(filepath)
                img_res = cv2.resize(img, RESOLUTION)
                cv2.imwrite(os.path.join(OUTPUT_DATASET, filename), img_res)
            except:
                print("File corrupt: " + filepath)


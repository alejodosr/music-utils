import cv2
import numpy as np


def apply_mask(image, mask, thr_body=95, alpha=1.0):
    mask = mask.astype(np.uint8)

    only_person = cv2.bitwise_and(image, image, mask=mask)

    lab = cv2.cvtColor(only_person, cv2.COLOR_BGR2LAB)

    # -----Splitting the LAB image to different channels-------------------------
    l, a, b = cv2.split(lab)

    # -----Applying CLAHE to L-channel-------------------------------------------
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(5, 5))
    cl = clahe.apply(l)
    ret, only_person_clahe = cv2.threshold(cl, thr_body, 255, cv2.THRESH_BINARY)

    # Convert to BGR
    only_person_clahe = cv2.cvtColor(only_person_clahe, cv2.COLOR_GRAY2BGR)

    # Add weighted sum
    only_person_final = cv2.addWeighted(only_person, alpha, only_person_clahe, 1 - alpha, gamma=0)

    # Segment background
    mask2 = 1 - mask
    image = cv2.bitwise_and(image, image, mask=mask2)

    # Join images
    image = cv2.add(image, only_person_final)

    return image
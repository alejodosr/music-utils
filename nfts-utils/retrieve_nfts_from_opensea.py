import requests
import json
import cv2
import os
import numpy as np
import time

TOTAL_NUM_IMAGES = 50000
ROOT_FOLDER = '/home/alejandro/temp'

dataset_folder = os.path.join(ROOT_FOLDER, 'ai_kitties')
os.system('mkdir -p ' + dataset_folder)

url = "https://api.opensea.io/api/v1/assets"

querystring = {"order_direction":"desc","offset":"101","limit":"150","collection": "cryptokitties"}

# print(len(json_response['assets']))

def address_to_img(address):
    adr = address.replace('0x', '')
    adr_bytes = bytes.fromhex(adr)
    # print(len(adr_bytes))
    img_np = np.zeros((255 * 256), np.uint8)
    for idx, letter in enumerate(adr_bytes):
        for jdx in range(255 * 256 // 20):
            img_np[idx * (255 * 256 // 20) + jdx] = letter
        # print(letter)
    img_np.shape = (255, 256)
    img_out = cv2.resize(img_np, (256, 256))
    # cv2.imshow('image', img_out)
    # cv2.waitKey(0)
    return img_out

def url_to_image(url):
    try:
        resp = requests.get(url, stream=True).raw
        image = np.asarray(bytearray(resp.read()), dtype="uint8")
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        image = cv2.resize(image, (256, 256))
        return image
    except:
        print("ERROR: Error in url image (" + url + ")")
        return None


num_images = 0
for i in range(TOTAL_NUM_IMAGES // 50):
    querystring["offset"] = i * 50
    querystring["limit"] = i * 50 + 50
    wrong = True
    while (wrong):
        try:
            response = requests.request("GET", url, params=querystring)
            json_response = json.loads(response.text)
            num_images += len(json_response['assets'])
            wrong = False
        except:
            wrong = True
            print("ERROR: Error in request, retrying in 1 sec...")
            time.sleep(1)

    for asset in json_response['assets']:
        # if asset['creator']['address'] is not None:
        #     adr_img = address_to_img(asset['creator']['address'])
        kitty_img = url_to_image(asset['image_url'])
        # cv2.imshow('image', kitty_img)
        # cv2.waitKey(0)
        if kitty_img is not None:
            cv2.imwrite(os.path.join(dataset_folder, str(asset['id']) + ".jpg"), kitty_img)
            # cv2.imwrite(os.path.join(dataset_folder, str(asset['id']) + ".png"), adr_img)
        # print(asset['image_url'])
        # print(asset['creator']['address'])
        # print(asset)
print(num_images)
# print(json_response['assets'])
# print(response.text)
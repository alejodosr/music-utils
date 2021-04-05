from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options

import os
import wget
import time
import subprocess

def convert_with_cairosvg_sizes(file_svg, file_png):
    from cairosvg.surface import PNGSurface
    with open(file_svg, 'rb') as svg_file:
        PNGSurface.convert(
            bytestring=svg_file.read(),
            width=512,
            height=512,
            write_to=open(file_png, 'wb')
            )


DATASET_FOLDER = '/media/alejandro/TOSHIBA EXT/ale/ai_kitties_90000'
DATASET_SIZE = 50000

chrome_options = Options()
chrome_options.add_argument("--headless")

url = "https://www.cryptokitties.co/search/"

# Initiate the browser
browser = webdriver.Chrome(ChromeDriverManager().install(), options=chrome_options)

# Number of images
imgs_num = 4007
start = imgs_num // 12

for i in range(start, DATASET_SIZE // 12):
    # Current numner of images
    print("INFO: Current number of images " + str(imgs_num))

    # Page number
    browser.get(url + str(i))
    print("INFO: Current url " + url + str(i))

    while len(browser.find_elements_by_tag_name('img')) <= 3:
        time.sleep(1)
        print("INFO: waiting to load page...")

    imgs = browser.find_elements_by_tag_name('img')
    for img in imgs:
        try:
            print("INFO: raw image svg " + str(img.get_attribute('src')))
            svg_filename = str(wget.download(img.get_attribute('src'), out=DATASET_FOLDER))

            convert_with_cairosvg_sizes(svg_filename, os.path.join(DATASET_FOLDER, svg_filename.split('/')[-1].split('.')[0] + ".png"))

            os.system("rm " + '"' + svg_filename + '"')

            # Increase the number of images
            imgs_num += 1
        except Exception as e:
            print("WARNING: Forbidden image")
            print(e)

    time.sleep(2)

browser.quit()

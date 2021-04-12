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


DATASET_FOLDER = '/media/alejandro/TOSHIBA EXT/ale/ai_kitties_40000'
DATASET_SIZE = 70000
MAX_NUMBER_OF_GENERATIONS = 30

chrome_options = Options()
chrome_options.add_argument("--headless")

url = "https://www.cryptokitties.co/search?search=gen:"

# Initiate the browser
browser = webdriver.Chrome(ChromeDriverManager().install(), options=chrome_options)

# Number of images
total_imgs_num = 22760
total_imgs_repeated = 16624
start = (total_imgs_num + total_imgs_repeated) // 12
# start_gen = (start // (DATASET_SIZE // (12 * MAX_NUMBER_OF_GENERATIONS)) + 1)
start_gen = 10
start = start // start_gen
print(start_gen)
print(start)
print(DATASET_SIZE // (12 * MAX_NUMBER_OF_GENERATIONS))

for gen in range(start_gen, MAX_NUMBER_OF_GENERATIONS):
    # Page number
    browser.get(url + str(gen))

    while len(browser.find_elements_by_class_name('Loader')):
        time.sleep(1)
        print("INFO: waiting to load page...")

    # Number of images per gen
    imgs_num = 0

    while imgs_num < (DATASET_SIZE // MAX_NUMBER_OF_GENERATIONS):
        # Current numner of images
        print("INFO: Current url " + url + str(gen))
        print("INFO: Generation " + str(gen))
        print("INFO: Current number of images " + str(total_imgs_num))
        print("INFO: Current redundant images " + str(total_imgs_repeated))
        print("INFO: Percentage of redundant images " + str(int(total_imgs_repeated / (1e-9 + total_imgs_num) * 100)) + "%")
        print("INFO: Target number of images per generation " + str(DATASET_SIZE // MAX_NUMBER_OF_GENERATIONS))

        imgs = browser.find_elements_by_tag_name('img')
        for img in imgs:
            try:
                print("INFO: raw image svg " + str(img.get_attribute('src')))

                if os.path.exists(os.path.join(DATASET_FOLDER, img.get_attribute('src').split('/')[-1].split('.')[0] + ".png")):
                    print("\nINFO: Redundant file: " + os.path.join(DATASET_FOLDER, img.get_attribute('src').split('/')[-1].split('.')[0] + ".png"))
                    total_imgs_repeated += 1
                else:
                    svg_filename = str(wget.download(img.get_attribute('src'), out=DATASET_FOLDER))
                    convert_with_cairosvg_sizes(svg_filename, os.path.join(DATASET_FOLDER, svg_filename.split('/')[-1].split('.')[0] + ".png"))
                    # Increase the number of images
                    total_imgs_num += 1
                    imgs_num += 1
                    # Remove svg
                    os.system("rm " + '"' + svg_filename + '"')

            except Exception as e:
                print("WARNING: Forbidden image")
                print(e)

        button = browser.find_elements_by_class_name('Pagination-button')[1]
        button.click()

        while len(browser.find_elements_by_class_name('Loader')):
            time.sleep(1)
            print("INFO: waiting to load page...")

        time.sleep(2)

browser.quit()

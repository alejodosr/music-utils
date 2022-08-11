import tempfile
import requests
from pathlib import Path
import os

def download_files_from_github(path, model_name):
    if model_name == "u2net":
        part1 = tempfile.NamedTemporaryFile(delete=False)
        part2 = tempfile.NamedTemporaryFile(delete=False)
        part3 = tempfile.NamedTemporaryFile(delete=False)
        part4 = tempfile.NamedTemporaryFile(delete=False)
        try:
            os.makedirs("/tmp/.u2net")
        except:
            print("u2net folder made or already exists")
        if not os.path.isfile(path):
            try:
                print('download part1 of %s' % model_name)
                part1_content = requests.get('https://github.com/nadermx/backgroundremover/raw/main/models/u2aa')
                part1.write(part1_content.content)
                part1.close()
                print('finished downloading part 1 of %s' % model_name)
                print('download part2 of %s' % model_name)

                part2_content = requests.get('https://github.com/nadermx/backgroundremover/raw/main/models/u2ab')
                part2.write(part2_content.content)
                part2.close()
                print('finished downloading part 2 of %s' % model_name)
                print('download part2 of %s' % model_name)

                part3_content = requests.get('https://github.com/nadermx/backgroundremover/raw/main/models/u2ac')
                part3.write(part3_content.content)
                part3.close()
                print('finished downloading part 3 of %s' % model_name)
                print('download part4 of %s' % model_name)

                part4_content = requests.get('https://github.com/nadermx/backgroundremover/raw/main/models/u2ad')
                part4.write(part4_content.content)
                part4.close()
                print('finished downloading part 4 of %s' % model_name)

                # sp.run(["cat", part1.name, part2.name, part3.name, part4.name, ">", path], stdout=sp.DEVNULL)
                # os.system(f'cat {part1.name} {part2.name} {part3.name} {part4.name} > {path}')
                with open(path, "wb") as file_object:
                    file_object.write(part1_content.content)
                with open(path, "ab") as file_object:
                    file_object.write(part2_content.content)
                    file_object.write(part3_content.content)
                    file_object.write(part4_content.content)
            finally:
                os.remove(part1.name)
                os.remove(part2.name)
                os.remove(part3.name)
                os.remove(part4.name)
    if model_name == "u2net_human_seg":
        part1 = tempfile.NamedTemporaryFile(delete=False)
        part2 = tempfile.NamedTemporaryFile(delete=False)
        part3 = tempfile.NamedTemporaryFile(delete=False)
        part4 = tempfile.NamedTemporaryFile(delete=False)
        if not os.path.isfile(path):
            try:
                print('download part1 of %s' % model_name)
                part1_content = requests.get('https://github.com/nadermx/backgroundremover/raw/main/models/u2haa')
                part1.write(part1_content.content)
                part1.close()
                print('finished downloading part 1 of %s' % model_name)
                print('download part2 of %s' % model_name)

                part2_content = requests.get('https://github.com/nadermx/backgroundremover/raw/main/models/u2hab')
                part2.write(part2_content.content)
                part2.close()
                print('finished downloading part 2 of %s' % model_name)
                print('download part2 of %s' % model_name)

                part3_content = requests.get('https://github.com/nadermx/backgroundremover/raw/main/models/u2hac')
                part3.write(part3_content.content)
                part3.close()
                print('finished downloading part 3 of %s' % model_name)
                print('download part4 of %s' % model_name)

                part4_content = requests.get('https://github.com/nadermx/backgroundremover/raw/main/models/u2had')
                part4.write(part4_content.content)
                part4.close()
                print('finished downloading part 4 of %s' % model_name)

                # sp.run(["cat", part1.name, part2.name, part3.name, part4.name, ">", path], stdout=sp.DEVNULL)
                # os.system(f'cat {part1.name} {part2.name} {part3.name} {part4.name} > {path}')
                with open(path, "wb") as file_object:
                    file_object.write(part1_content.content)
                with open(path, "ab") as file_object:
                    file_object.write(part2_content.content)
                    file_object.write(part3_content.content)
                    file_object.write(part4_content.content)
            finally:
                os.remove(part1.name)
                os.remove(part2.name)
                os.remove(part3.name)
                os.remove(part4.name)

    if model_name == "u2netp":
        part1 = tempfile.NamedTemporaryFile(delete=False)
        try:
            print('download %s' % model_name)
            part1_content = requests.get('https://github.com/nadermx/backgroundremover/raw/main/models/u2haa')
            part1.write(part1_content.content)
            part1.close()
            print('finished downloading %s' % model_name)
            sp.run(["cat", part1.name, ">", path], stdout=sp.DEVNULL)
        finally:
            os.remove(part1.name)

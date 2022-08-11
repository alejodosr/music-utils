import os
import urllib.request
from random import randrange
import re
import pytube

# Keywords
KEYWORDS = ['Dilinyer', 'Neutro+Shorty', 'Lil+Mexico', '6IX9INE', '970BLOCK', 'RICH+FLAMZ', 'Anuel+AA',
            'Lucho+SSJ', 'ISRAEL+B', 'AKUNA', 'Mora+x+Juliito', 'MC+Buzzz', 'Robgz', 'KEVVO']

WORKING_DIR = '/home/alejandro/temp/test_youtube'
os.system('mkdir -p ' + os.path.join(WORKING_DIR, 'segments'))
os.system('mkdir -p ' + os.path.join(WORKING_DIR, 'images'))

# Download random video
number_of_segment = 1
html = urllib.request.urlopen("https://www.youtube.com/results?search_query=" + KEYWORDS[randrange(len(KEYWORDS))].lower())
video_ids = re.findall(r"watch\?v=(\S{11})", html.read().decode())
youtube = pytube.YouTube("https://www.youtube.com/watch?v=" + video_ids[0])
video_path = youtube.streams.filter(res="720p").first().download(os.path.join(WORKING_DIR, 'segments'))
video_root = video_path
os.system('mv "' + video_path + '" ' + video_root.replace(video_path.split('/')[-1], '') + 'video_' + str(number_of_segment).zfill(2) + ".mp4")

# Segment start random between start and end of video
# Size random between 0.2 and 0.5
# Segment length randomized between 2-10 sec
# center position: random between 1920x10080 margins
# Max. number of videos per frame 5

def download_from_youtube(url):
    youtube = pytube.YouTube(url)
    return youtube.streams.filter(res="720p").first().download('/tmp')

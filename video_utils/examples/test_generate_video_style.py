import os

from video_utils.generate_video_democracy import generate_video_style

video_output = generate_video_style('/home/alejandro/temp/test_ale.mp4', style='fresssh', substyle=None)

os.system(f'mv {video_output} {os.path.join("/tmp", os.path.basename(video_output))}')

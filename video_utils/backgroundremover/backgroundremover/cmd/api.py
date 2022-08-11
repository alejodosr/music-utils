import argparse
import os
from distutils.util import strtobool
from video_utils.backgroundremover.backgroundremover.utilities import *
from video_utils.backgroundremover.backgroundremover.bg import remove


def process_video(
    input,
    backgroundvideo='',
    backgroundimage='',
    model='u2net',
    alpha_matting=False,
    alpha_matting_foreground_threshold=240,
    alpha_matting_background_threshold=10,
    alpha_matting_erode_size=10,
    alpha_matting_base_size=1000,
    workernodes=1,
    gpubatchsize=2,
    framerate=-1,
    framelimit=-1,
    type='greenvideo'  # mattekey, transparentvideo, greenvideo, transparentvideoovervideo,
    # transparentvideooverimage, transparentgif, transparentgifwithbackground, backgroundimage, backgroundvideo
):
    model_choices = ["u2net", "u2net_human_seg", "u2netp"]

    output = '/tmp/greenvideo.mp4'  # or mov?

    if input.rsplit('.', 1)[1] in ['mp4', 'mov', 'webm', 'ogg', 'gif']:
        if type == 'mattekey':
            matte_key(os.path.abspath(output), os.path.abspath(input),
                                worker_nodes=workernodes,
                                gpu_batchsize=gpubatchsize,
                                model_name=model,
                                frame_limit=framelimit,
                                framerate=framerate)
        elif type == 'transparentvideo':
            transparentvideo(os.path.abspath(output), os.path.abspath(input),
                                       worker_nodes=workernodes,
                                       gpu_batchsize=gpubatchsize,
                                       model_name=model,
                                       frame_limit=framelimit,
                                       framerate=framerate)
        elif type == 'greenvideo':
            greenvideo(os.path.abspath(output), os.path.abspath(input),
                                 worker_nodes=workernodes,
                                 gpu_batchsize=gpubatchsize,
                                 model_name=model,
                                 frame_limit=framelimit,
                                 framerate=framerate)
        elif type == 'transparentvideoovervideo':
            transparentvideoovervideo(os.path.abspath(output),
                                                os.path.abspath(backgroundvideo),
                                                os.path.abspath(input),
                                                worker_nodes=workernodes,
                                                gpu_batchsize=gpubatchsize,
                                                model_name=model,
                                                frame_limit=framelimit,
                                                framerate=framerate)
        elif type == 'transparentvideooverimage':
            transparentvideooverimage(os.path.abspath(output),
                                                os.path.abspath(backgroundimage),
                                                os.path.abspath(input),
                                                worker_nodes=workernodes,
                                                gpu_batchsize=gpubatchsize,
                                                model_name=model,
                                                frame_limit=framelimit,
                                                framerate=framerate)
        elif type == 'transparentgif':
            transparentgif(os.path.abspath(output), os.path.abspath(input),
                                     worker_nodes=workernodes,
                                     gpu_batchsize=gpubatchsize,
                                     model_name=model,
                                     frame_limit=framelimit,
                                     framerate=framerate)
        elif type == 'transparentgifwithbackground':
            transparentgifwithbackground(os.path.abspath(output),
                                                   os.path.abspath(backgroundimage),
                                                   os.path.abspath(input),
                                                   worker_nodes=workernodes,
                                                   gpu_batchsize=gpubatchsize,
                                                   model_name=model,
                                                   frame_limit=framelimit,
                                                   framerate=framerate)

    else:
        print(output)
        r = lambda i: i.buffer.read() if hasattr(i, "buffer") else i.read()
        w = lambda o, data: o.buffer.write(data) if hasattr(o, "buffer") else o.write(data)
        w(
            output,
            remove(
                r(input),
                model_name=model,
                alpha_matting=alpha_matting,
                alpha_matting_foreground_threshold=alpha_matting_foreground_threshold,
                alpha_matting_background_threshold=alpha_matting_background_threshold,
                alpha_matting_erode_structure_size=alpha_matting_erode_size,
                alpha_matting_base_size=alpha_matting_base_size,
            ),
        )

    return output

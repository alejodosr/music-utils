import argparse

from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import os, cv2
import face_detection

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils.visualizer import ColorMode
import numpy as np
import glob
import os
from PIL import Image


from video_utils.person_changers.bansky_person_handlers import apply_mask
from video_utils.face_changers.anime_face_changer import AnimeFaceChanger
from video_utils.face_changers.dual_stylegan_face_changer import DualStyleGan2FaceChanger
from video_utils.backgroundremover.backgroundremover.cmd.api import process_video


def alpha_blend(img1, img2, mask_thickness=(0.14, 0)):
    h, w = img1.shape[:2]
    blend_mask = np.zeros((h, w), np.uint8)
    cv2.rectangle(blend_mask, (0, h - int(mask_thickness[0] * h)), (w, h), (255, 255, 255), -1, cv2.LINE_AA)
    blend_mask = cv2.GaussianBlur(blend_mask, (21, 21), 21)

    if blend_mask.ndim == 3 and blend_mask.shape[-1] == 3:
        alpha = blend_mask/255.0
    else:
        alpha = cv2.cvtColor(blend_mask, cv2.COLOR_GRAY2BGR)/255.0

    blended = cv2.convertScaleAbs(img2 * (1 - alpha) + img1 * alpha)
    return blended

def dilate_face_mask(face_mask, dilation_rate):
    dilated_size = (int((ymax - ymin) * dilation_rate[0]), int((xmax - xmin) * dilation_rate[0]))
    face_mask_dilated = cv2.dilate(face_mask.copy(), np.ones(dilated_size), iterations=1)  # (y, x)
    original_rate = 0.75
    face_mask[0:int(original_rate * ymax), 0:xmax] = face_mask_dilated[0:int(original_rate * ymax), 0:xmax]
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (51, 51))
    return cv2.morphologyEx(face_mask, cv2.MORPH_OPEN, kernel, iterations=5)

def get_human_masks(img, predictor):
    outputs = predictor(img)

    pred_classes = outputs["instances"].pred_classes.cpu().numpy()
    masks = outputs["instances"].pred_masks.cpu()

    masks_out = []
    for idx, pred_class in enumerate(pred_classes):
        if pred_class == 0:
            mask = masks[idx, :, :].squeeze(0).numpy().astype(np.uint8)
            mask = mask.astype(np.uint8)

            masks_out.append(mask)

    return masks_out

def make_video(outvid, images=None, fps=30, size=None,
               is_color=True, format="FMP4"):
    """
    Create a video from a list of images.

    @param      outvid      output video
    @param      images      list of images to use in the video
    @param      fps         frame per second
    @param      size        size of each frame
    @param      is_color    color
    @param      format      see http://www.fourcc.org/codecs.php
    @return                 see http://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_gui/py_video_display/py_video_display.html

    The function relies on http://opencv-python-tutroals.readthedocs.org/en/latest/.
    By default, the video will have the size of the first image.
    It will resize every image to this size before adding them to the video.
    """
    from cv2 import VideoWriter, VideoWriter_fourcc, imread, resize
    fourcc = VideoWriter_fourcc(*format)
    vid = None
    for image in images:
        if not os.path.exists(image):
            raise FileNotFoundError(image)
        img = imread(image)
        if vid is None:
            if size is None:
                size = img.shape[1], img.shape[0]
            vid = VideoWriter(outvid, fourcc, float(fps), size, is_color)
        if size[0] != img.shape[1] and size[1] != img.shape[0]:
            img = resize(img, size)
        vid.write(img)
    vid.release()
    return vid


def generate_photo_style(input_photo_path,
                         model='detectron'  # detectron, unet
                         ):

    if model == 'detectron':
        cfg = get_cfg()
        # Inference should use the config with parameters that are used in training
        # cfg now already contains everything we've set previously. We changed it a little bit for inference:
        cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        cfg.DATASETS.TEST = ()
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
            "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
        cfg.TEST.DETECTIONS_PER_IMAGE = 1000
        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128  # faster, and good enough for this toy dataset (default: 512)
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.1  # set a custom testing threshold

    if model == 'detectron':
        # Human segmentation
        predictor = DefaultPredictor(cfg)

    frame = cv2.imread(input_photo_path)
    masks_out = get_human_masks(frame, predictor)

    imgs_out = []
    thresholds = [40, 80, 95, 105, 125, 150, 200]
    for thr  in thresholds:
        img_demo = frame.copy()
        for mask in masks_out:
            if mask.shape[:2] != img_demo.shape[:2]:
                mask = cv2.resize(mask, (img_demo.shape[1], img_demo.shape[0]), interpolation=cv2.INTER_AREA)
                mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
                (T, mask) = cv2.threshold(mask, 127, 255,
                                          cv2.THRESH_BINARY)
                mask = cv2.normalize(mask, dst=None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

            img_demo = apply_mask(img_demo, mask, thr_body=thr)

        output_path = f'/tmp/photo_out_{thr}.jpg'

        cv2.imwrite(output_path, img_demo)

        imgs_out.append(output_path)

    return imgs_out



def generate_video_style(input_video,
                         style,
                         substyle,
                         resegment=True,
                         audio=True,
                         model='detectron'   # unet, detectron
                         ):
    if os.path.isdir('save'):
        os.system(f'rm -rf ./save')

    global xmin, ymin, xmax, ymax, size, video
    if style == 'anime':
        dilation_rate = (0.2, 0.15)
        face_changer = AnimeFaceChanger()
    elif style == 'dual':
        dilation_rate = (0.4, 0.4)
        face_changer = DualStyleGan2FaceChanger(download=True, style=int(substyle))
    elif style == 'fresssh':
        pass
    else:
        raise Exception("Style not defined")

    # create output folder
    os.system('mkdir save')

    if model == 'detectron':
        cfg = get_cfg()
        # Inference should use the config with parameters that are used in training
        # cfg now already contains everything we've set previously. We changed it a little bit for inference:
        cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        cfg.DATASETS.TEST = ()
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
            "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
        cfg.TEST.DETECTIONS_PER_IMAGE = 1000
        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128  # faster, and good enough for this toy dataset (default: 512)
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.1  # set a custom testing threshold

    elif model == 'unet':
        mask_video = process_video(input_video, type='mattekey')
        cap_mask = cv2.VideoCapture(mask_video)
        mask_frames = []
        while cap_mask.isOpened():
            ret, frame = cap_mask.read()
            if ret:
                mask_frames.append(frame)
            else:
                break

    detector = face_detection.build_detector(
        "DSFDDetector", confidence_threshold=.5, nms_iou_threshold=.3)

    # Generate fresssh
    cap = cv2.VideoCapture(input_video)

    if cap.isOpened() == False:
        print("Error opening video stream or file")
    # Read until video is completed
    # cap.set(1, 500)
    frame_count = 0
    while cap.isOpened():
        # Capture frame-by-frame
        ret, frame = cap.read()

        if ret:

            # Add brightness
            # frame = cv2.add(frame, np.array([50.0]))
            # print("INFO: Adding brightness")

            frame_count += 1

            # if frame_count <= 150:
            #     continue

            if model == 'detectron':
                # Human segmentation
                predictor = DefaultPredictor(cfg)
                masks_out = get_human_masks(frame, predictor)
            elif model == 'unet':
                masks_out = [mask_frames[frame_count - 1]]

            img_demo = frame
            for mask in masks_out:
                if mask.shape[:2] != img_demo.shape[:2]:
                    mask = cv2.resize(mask, (img_demo.shape[1], img_demo.shape[0]), interpolation=cv2.INTER_AREA)
                    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
                    (T, mask) = cv2.threshold(mask, 127, 255,
                                                   cv2.THRESH_BINARY)
                    mask = cv2.normalize(mask, dst=None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

                img_demo = apply_mask(img_demo, mask)

                # Face detector
                detections = detector.detect(frame[:, :, ::-1])
                if len(detections):
                    face_bbox = [int(value) for value in detections[0]]
                    h, w, c = frame.shape
                    xmin, ymin, xmax, ymax, conf = face_bbox
                    padding = 900 * (xmax - xmin) / w
                    xmin = max(0, int(xmin - padding))
                    ymin = max(0, int(ymin - padding))
                    xmax = min(w, int(xmax + padding))
                    ymax = min(h, int(ymax + padding))

                    if style is not None and style != 'fresssh':
                        in_img = frame[ymin:ymax, xmin:xmax, :]
                        size = (512, 512)
                        resized_in_img = cv2.resize(in_img.copy(), size, interpolation=cv2.INTER_AREA)
                        out_img_pil = face_changer.change_face(Image.fromarray(cv2.cvtColor(resized_in_img,
                                                                                            cv2.COLOR_BGR2RGB)))

                        if out_img_pil is not None:
                            out_img = np.array(out_img_pil)
                            # Convert RGB to BGR
                            out_img = out_img[:, :, ::-1].copy()
                            out_img = cv2.resize(out_img, (xmax - xmin, ymax - ymin), interpolation=cv2.INTER_AREA)

                            face_mask = mask[ymin:ymax, xmin:xmax].copy()

                            # Improve face mask due to anime distortion of original sizes
                            modified_face_mask = face_mask.copy()

                            if resegment and model != 'unet':
                                masks_out = get_human_masks(out_img, predictor)
                                modified_face_mask = masks_out[0] if len(masks_out) else dilate_face_mask(
                                    modified_face_mask, dilation_rate)

                            face_mask = cv2.bitwise_or(face_mask, modified_face_mask)

                            masked_img = cv2.bitwise_and(out_img, out_img, mask=face_mask)

                            # Segment background
                            face_mask_inverse = 1 - face_mask
                            image_back = cv2.bitwise_and(in_img, in_img, mask=face_mask_inverse)

                            final_face_img = cv2.add(masked_img, image_back)

                            img_demo[ymin:ymax, xmin:xmax, :] = \
                                alpha_blend(img_demo[ymin:ymax, xmin:xmax, :], final_face_img)

                    if mask[ymin:ymax, xmin:xmax].sum() < 200:  # Detected face with mask rcnn
                        mask[ymin:ymax, xmin:xmax] = np.ones((ymax - ymin, xmax - xmin))

            print('./save/' + str(frame_count).zfill(8) + '.jpg')
            cv2.imwrite('./save/' + str(frame_count).zfill(8) + '.jpg', img_demo)

            # cv2.namedWindow("img", cv2.WINDOW_NORMAL)        # Create window with freedom of dimensions
            # cv2.imshow('img', img_demo)
            # cv2.waitKey(0)
        else:
            # When everything done, release the video capture object
            cap.release()
    # Closes all the frames
    cv2.destroyAllWindows()
    # Generate video
    video = cv2.VideoCapture(input_video)
    # Find OpenCV version
    (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
    if int(major_ver) < 3:
        fps = video.get(cv2.cv.CV_CAP_PROP_FPS)
        print("Frames per second using video.get(cv2.cv.CV_CAP_PROP_FPS): {0}".format(fps))
    else:
        fps = video.get(cv2.CAP_PROP_FPS)
        print("Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps))
    video.release()
    # Directory of images to run detection on
    VIDEO_SAVE_DIR = "save"
    images = list(glob.iglob(os.path.join(VIDEO_SAVE_DIR, '*.*')))
    # Sort the images by integer index
    images = sorted(images, key=lambda x: float(os.path.split(x)[1][:-3]))
    outvid = f"./save/out_{style}.mp4"

    if not audio:
        # Make video with OpenCV
        make_video(outvid, images, fps=fps)
    else:
        # Make video with audio
        os.system(f'ffmpeg -i {input_video} -vn -acodec copy ./save/output-audio.aac')
        os.system(f'ffmpeg -r {fps:0.2f} -start_number 1 -i ./save/%08d.jpg -i ./save/output-audio.aac -preset slow '
                  f'-c:a aac -b:a 128k -codec:v libx264 -pix_fmt yuv420p -b:v 2500k '
                  f'-minrate 1500k -maxrate 4000k -bufsize 5000k {outvid}')


        # Other for audio
        # ffmpeg -i /home/alejandro/fresssh/releases/20221221_woman/video_carlos/extas_bien_v1.mov -map 0:a -c copy output-audio.mov
        # High Quality rendering. Play with higher values of crf to increase quality vs size (lower higher quality)
        # ffmpeg -r 29.97 -start_number 1 -i ./%08d.jpg -i ./output-audio.mov -c:v libx264 -crf 12 -preset veryslow -c:a aac estas_bien_fresssh_v1.mp4


    return os.path.abspath(outvid)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", type=str,
                    help="input video to process",
                    default='/home/alejandro/fresssh/releases/20221221_woman/video_carlos/extas_bien_v1.mov')
    ap.add_argument("-s", "--style", type=str,
                    help="'anime', 'dual0', 'dual1",
                    default='fresssh')
    ap.add_argument("-r", "--resegment", action='store_true',
                    help="Resegment after face change",
                    default=True)

    args = vars(ap.parse_args())

    # Read video
    input_video = args['input']
    style = args['style'][:-1] if 'dual' in args['style'] else args['style']
    substyle = args['style'][-1]
    resegment = args['resegment']

    generate_video_style(input_video, style, substyle, resegment)

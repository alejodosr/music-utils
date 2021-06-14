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


def apply_mask(image, mask, thr_body=95):
    mask = mask.astype(np.uint8)

    only_person = cv2.bitwise_and(image, image, mask=mask)

    lab = cv2.cvtColor(only_person, cv2.COLOR_BGR2LAB)

    # -----Splitting the LAB image to different channels-------------------------
    l, a, b = cv2.split(lab)

    # -----Applying CLAHE to L-channel-------------------------------------------
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(1, 1))
    cl = clahe.apply(l)
    ret, only_person = cv2.threshold(cl, thr_body, 255, cv2.THRESH_BINARY)

    # Segment background
    mask2 = 1 - mask
    image = cv2.bitwise_and(image, image, mask=mask2)

    # Convert to BGR
    only_person = cv2.cvtColor(only_person, cv2.COLOR_GRAY2BGR)

    # Join images
    image = cv2.add(image, only_person)

    return image

# Read video
INPUT_VIDEO = 'cambia.mov'

#Generate democracy
cap = cv2.VideoCapture(INPUT_VIDEO)

# create output folder
os.system('mkdir save')

cfg = get_cfg()
# Inference should use the config with parameters that are used in training
# cfg now already contains everything we've set previously. We changed it a little bit for inference:


cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))

cfg.DATASETS.TEST = ()
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
cfg.TEST.DETECTIONS_PER_IMAGE = 1000
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.1   # set a custom testing threshold

detector = face_detection.build_detector(
  "DSFDDetector", confidence_threshold=.5, nms_iou_threshold=.3)

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
        frame = cv2.add(frame, np.array([50.0]))
        print("INFO: Adding brightness")

        frame_count += 1

        predictor = DefaultPredictor(cfg)
        outputs = predictor(frame)

        pred_classes = outputs["instances"].pred_classes.cpu().numpy()
        masks = outputs["instances"].pred_masks.cpu()

        img_demo = frame
        for idx, pred_class in enumerate(pred_classes):
            if pred_class == 0:
                mask = masks[idx, :, :].squeeze(0).numpy()

                # Face detector
                detections = detector.detect(frame[:, :, ::-1])
                if len(detections):
                    face_bbox = [int(value) for value in detections[0]]
                    xmin, ymin, xmax, ymax, conf = face_bbox
                    print(face_bbox)

                    # if mask[ymin:ymax, xmin:xmax].sum() < 200:  # Detected face with mask rcnn
                    mask[ymin:ymax, xmin:xmax] = np.ones((ymax - ymin, xmax - xmin))

                img_demo = apply_mask(img_demo, mask)

        print('./save/' + str(frame_count).zfill(8) + '.jpg')
        cv2.imwrite('./save/' + str(frame_count).zfill(8) + '.jpg', img_demo)

        # cv2.namedWindow("img", cv2.WINDOW_NORMAL)        # Create window with freedom of dimensions
        # cv2.imshow('img', img_demo)
        # cv2.waitKey(1)
    else:
        # When everything done, release the video capture object
        cap.release()

# Closes all the frames
cv2.destroyAllWindows()

# Generate video
video = cv2.VideoCapture(INPUT_VIDEO)

# Find OpenCV version
(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

if int(major_ver)  < 3 :
    fps = video.get(cv2.cv.CV_CAP_PROP_FPS)
    print("Frames per second using video.get(cv2.cv.CV_CAP_PROP_FPS): {0}".format(fps))
else :
    fps = video.get(cv2.CAP_PROP_FPS)
    print("Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps))

video.release()


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


# Directory of images to run detection on
VIDEO_SAVE_DIR = "save"
images = list(glob.iglob(os.path.join(VIDEO_SAVE_DIR, '*.*')))
# Sort the images by integer index
images = sorted(images, key=lambda x: float(os.path.split(x)[1][:-3]))

outvid = "./out.mp4"
make_video(outvid, images, fps=fps)
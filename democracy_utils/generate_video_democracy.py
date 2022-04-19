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


from person_changers.bansky_person_handlers import apply_mask
from face_changers.anime_face_changer import AnimeFaceChanger


def alpha_blend(img1, img2, mask_thickness=(0.2, 0)):
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

# Read video
INPUT_VIDEO = '/home/alejandro/fresssh/tests/encoder.mp4'
ANIME_FACE = True

if ANIME_FACE:
    anime_face_changer = AnimeFaceChanger()

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
        # frame = cv2.add(frame, np.array([50.0]))
        # print("INFO: Adding brightness")

        frame_count += 1

        predictor = DefaultPredictor(cfg)
        outputs = predictor(frame)

        pred_classes = outputs["instances"].pred_classes.cpu().numpy()
        masks = outputs["instances"].pred_masks.cpu()

        img_demo = frame
        for idx, pred_class in enumerate(pred_classes):
            if pred_class == 0:
                mask = masks[idx, :, :].squeeze(0).numpy().astype(np.uint8)
                mask = mask.astype(np.uint8)

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

                    if ANIME_FACE:
                        in_img = frame[ymin:ymax, xmin:xmax, :]
                        resized_in_img = cv2.resize(in_img.copy(), (512, 512), interpolation=cv2.INTER_AREA)
                        out_img_pil = anime_face_changer.change_face(Image.fromarray(cv2.cvtColor(resized_in_img,
                                                                      cv2.COLOR_BGR2RGB)))

                        out_img = np.array(out_img_pil)
                        # Convert RGB to BGR
                        out_img = out_img[:, :, ::-1].copy()
                        out_img = cv2.resize(out_img, (xmax - xmin, ymax - ymin), interpolation=cv2.INTER_AREA)

                        face_mask = mask[ymin:ymax, xmin:xmax].copy()

                        # Improve face mask due to anime distortion of original sizes
                        modified_face_mask = face_mask.copy()

                        dilation_rate = (0.2, 0.15)
                        dilated_size = (int((ymax - ymin) * dilation_rate[0]), int((xmax - xmin) * dilation_rate[0]))
                        face_mask_dilated = cv2.dilate(modified_face_mask.copy(),  np.ones(dilated_size), iterations=1)  # (y, x)
                        original_rate = 0.75
                        modified_face_mask[0:int(original_rate * ymax), 0:xmax] = face_mask_dilated[0:int(original_rate * ymax), 0:xmax]
                        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (51, 51))
                        modified_face_mask = cv2.morphologyEx(modified_face_mask, cv2.MORPH_OPEN, kernel, iterations=5)

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

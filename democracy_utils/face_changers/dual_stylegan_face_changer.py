# https://github.com/williamyang1991/DualStyleGAN
import os

import PIL
import cv2
import dlib
import numpy
import numpy as np
import scipy
import scipy.ndimage
import torch
from face_changers.dual_style_gan2.util import save_image, load_image, visualize
import argparse
from argparse import Namespace
from torchvision import transforms
from torch.nn import functional as F
import torchvision
import matplotlib.pyplot as plt
from face_changers.dual_style_gan2.model.dualstylegan import DualStyleGAN
from face_changers.dual_style_gan2.model.sampler.icp import ICPTrainer
from face_changers.dual_style_gan2.model.encoder.psp import pSp


class DualStyleGan2FaceChanger:
    def __init__(self, download=True, style=0):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.MODEL_DIR = 'checkpoint'
        self.DATA_DIR = 'face_changers/dual_style_gan2/data'
        assert os.path.isdir(self.MODEL_DIR)
        assert os.path.isdir(self.DATA_DIR)
        self.if_align_face = True
        self.style_types = ['cartoon', 'caricature', 'anime', 'arcane', 'comic', 'pixar', 'slamdunk']
        self.style_type = self.style_types[style]
        self.style_id = 26

        if not os.path.exists(os.path.join(self.MODEL_DIR, self.style_type)):
            os.makedirs(os.path.join(self.MODEL_DIR, self.style_type))

        MODEL_PATHS = {
            "encoder": {"id": "1NgI4mPkboYvYw3MWcdUaQhkr0OWgs9ej", "name": "encoder.pt"},
            "cartoon-G": {"id": "1exS9cSFkg8J4keKPmq2zYQYfJYC5FkwL", "name": "generator.pt"},
            "cartoon-N": {"id": "1JSCdO0hx8Z5mi5Q5hI9HMFhLQKykFX5N", "name": "sampler.pt"},
            "cartoon-S": {"id": "1ce9v69JyW_Dtf7NhbOkfpH77bS_RK0vB", "name": "refined_exstyle_code.npy"},
            "caricature-G": {"id": "1BXfTiMlvow7LR7w8w0cNfqIl-q2z0Hgc", "name": "generator.pt"},
            "caricature-N": {"id": "1eJSoaGD7X0VbHS47YLehZayhWDSZ4L2Q", "name": "sampler.pt"},
            "caricature-S": {"id": "1-p1FMRzP_msqkjndRK_0JasTdwQKDsov", "name": "refined_exstyle_code.npy"},
            "anime-G": {"id": "1BToWH-9kEZIx2r5yFkbjoMw0642usI6y", "name": "generator.pt"},
            "anime-N": {"id": "19rLqx_s_SUdiROGnF_C6_uOiINiNZ7g2", "name": "sampler.pt"},
            "anime-S": {"id": "17-f7KtrgaQcnZysAftPogeBwz5nOWYuM", "name": "refined_exstyle_code.npy"},
            "arcane-G": {"id": "15l2O7NOUAKXikZ96XpD-4khtbRtEAg-Q", "name": "generator.pt"},
            "arcane-N": {"id": "1fa7p9ZtzV8wcasPqCYWMVFpb4BatwQHg", "name": "sampler.pt"},
            "arcane-S": {"id": "1z3Nfbir5rN4CrzatfcgQ8u-x4V44QCn1", "name": "exstyle_code.npy"},
            "comic-G": {"id": "1_t8lf9lTJLnLXrzhm7kPTSuNDdiZnyqE", "name": "generator.pt"},
            "comic-N": {"id": "1RXrJPodIn7lCzdb5BFc03kKqHEazaJ-S", "name": "sampler.pt"},
            "comic-S": {"id": "1ZfQ5quFqijvK3hO6f-YDYJMqd-UuQtU-", "name": "exstyle_code.npy"},
            "pixar-G": {"id": "1TgH7WojxiJXQfnCroSRYc7BgxvYH9i81", "name": "generator.pt"},
            "pixar-N": {"id": "18e5AoQ8js4iuck7VgI3hM_caCX5lXlH_", "name": "sampler.pt"},
            "pixar-S": {"id": "1I9mRTX2QnadSDDJIYM_ntyLrXjZoN7L-", "name": "exstyle_code.npy"},
            "slamdunk-G": {"id": "1MGGxSCtyf9399squ3l8bl0hXkf5YWYNz", "name": "generator.pt"},
            "slamdunk-N": {"id": "1-_L7YVb48sLr_kPpOcn4dUq7Cv08WQuG", "name": "sampler.pt"},
            "slamdunk-S": {"id": "1Dgh11ZeXS2XIV2eJZAExWMjogxi_m_C8", "name": "exstyle_code.npy"},
        }

        # download pSp encoder
        path = MODEL_PATHS["encoder"]
        download_command = self.get_download_model_command(file_id=path["id"], file_name=path["name"])
        if download:
            os.system(download_command)

        # download dualstylegan
        path = MODEL_PATHS[self.style_type + '-G']
        download_command = self.get_download_model_command(file_id=path["id"],
                                                      file_name=os.path.join(self.style_type, path["name"]))
        if download:
            os.system(download_command)
        # download sampler
        path = MODEL_PATHS[self.style_type + '-N']
        download_command = self.get_download_model_command(file_id=path["id"],
                                                      file_name=os.path.join(self.style_type, path["name"]))
        if download:
            os.system(download_command)
        # download extrinsic style code
        path = MODEL_PATHS[self.style_type + '-S']
        download_command = self.get_download_model_command(file_id=path["id"],
                                                      file_name=os.path.join(self.style_type, path["name"]))
        if download:
            os.system(download_command)

        self.transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(256),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )

        # load DualStyleGAN
        generator = DualStyleGAN(1024, 512, 8, 2, res_index=6)
        generator.eval()
        ckpt = torch.load(os.path.join(self.MODEL_DIR, self.style_type, 'generator.pt'),
                          map_location=lambda storage, loc: storage)
        generator.load_state_dict(ckpt["g_ema"])
        self.generator = generator.to(self.device)

        # load encoder
        model_path = os.path.join(self.MODEL_DIR, 'encoder.pt')
        ckpt = torch.load(model_path, map_location='cpu')
        opts = ckpt['opts']
        opts['checkpoint_path'] = model_path
        opts = Namespace(**opts)
        opts.device = self.device
        encoder = pSp(opts)
        encoder.eval()
        self.encoder = encoder.to(self.device)

        # load extrinsic style code
        self.exstyles = np.load(os.path.join(self.MODEL_DIR, self.style_type, MODEL_PATHS[self.style_type + '-S']["name"]),
                           allow_pickle='TRUE').item()

        # load sampler network
        icptc = ICPTrainer(np.empty([0, 512 * 11]), 128)
        icpts = ICPTrainer(np.empty([0, 512 * 7]), 128)
        ckpt = torch.load(os.path.join(self.MODEL_DIR, self.style_type, 'sampler.pt'), map_location=lambda storage, loc: storage)
        icptc.icp.netT.load_state_dict(ckpt['color'])
        icpts.icp.netT.load_state_dict(ckpt['structure'])
        icptc.icp.netT = icptc.icp.netT.to(self.device)
        icpts.icp.netT = icpts.icp.netT.to(self.device)

        print('Model successfully loaded!')

    def get_download_model_command(self, file_id, file_name):
        """ Get wget download command for downloading the desired model and save to directory ../checkpoint/. """
        current_directory = os.getcwd()
        save_path = self.MODEL_DIR
        url = r"""wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id={FILE_ID}' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id={FILE_ID}" -O {SAVE_PATH}/{FILE_NAME} && rm -rf /tmp/cookies.txt""".format(
            FILE_ID=file_id, FILE_NAME=file_name, SAVE_PATH=save_path)
        return url

    def run_alignment(self, rgb_face_img):
        import dlib
        self.modelname = os.path.join(self.MODEL_DIR, 'shape_predictor_68_face_landmarks.dat')
        if not os.path.exists(self.modelname):
            import wget, bz2
            wget.download('http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2', self.modelname + '.bz2')
            zipfile = bz2.BZ2File(self.modelname + '.bz2')
            data = zipfile.read()
            open(self.modelname, 'wb').write(data)
        predictor = dlib.shape_predictor(self.modelname)
        aligned_image, crop, quad, transform_size = self.align_face(rgb_face_img, predictor=predictor)
        return aligned_image, crop, quad, transform_size

    def get_landmark(self, rgb_img, predictor):
        """get landmark with dlib
        :return: np.array shape=(68, 2)
        """
        detector = dlib.get_frontal_face_detector()

        # img = dlib.load_rgb_image(filepath)
        img = numpy.array(rgb_img)[:, :, ::-1]
        dets = detector(img, 1)

        for k, d in enumerate(dets):
            shape = predictor(img, d)

        t = list(shape.parts())
        a = []
        for tt in t:
            a.append([tt.x, tt.y])
        lm = np.array(a)
        return lm

    def align_face(self, rgb_img, predictor):
        """
        :param filepath: str
        :return: PIL Image
        """

        lm = self.get_landmark(rgb_img, predictor)

        lm_chin = lm[0: 17]  # left-right
        lm_eyebrow_left = lm[17: 22]  # left-right
        lm_eyebrow_right = lm[22: 27]  # left-right
        lm_nose = lm[27: 31]  # top-down
        lm_nostrils = lm[31: 36]  # top-down
        lm_eye_left = lm[36: 42]  # left-clockwise
        lm_eye_right = lm[42: 48]  # left-clockwise
        lm_mouth_outer = lm[48: 60]  # left-clockwise
        lm_mouth_inner = lm[60: 68]  # left-clockwise

        # Calculate auxiliary vectors.
        eye_left = np.mean(lm_eye_left, axis=0)
        eye_right = np.mean(lm_eye_right, axis=0)
        eye_avg = (eye_left + eye_right) * 0.5
        eye_to_eye = eye_right - eye_left
        mouth_left = lm_mouth_outer[0]
        mouth_right = lm_mouth_outer[6]
        mouth_avg = (mouth_left + mouth_right) * 0.5
        eye_to_mouth = mouth_avg - eye_avg

        # Choose oriented crop rectangle.
        x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
        x /= np.hypot(*x)
        x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
        y = np.flipud(x) * [-1, 1]
        c = eye_avg + eye_to_mouth * 0.1
        quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
        qsize = np.hypot(*x) * 2

        # read image
        # img = PIL.Image.fromarray(rgb_img)
        img = rgb_img

        # img = PIL.Image.open(filepath)

        output_size = 256
        transform_size = 256
        enable_padding = False

        # Shrink.
        shrink = int(np.floor(qsize / output_size * 0.5))
        if shrink > 1:
            rsize = (int(np.rint(float(img.size[0]) / shrink)), int(np.rint(float(img.size[1]) / shrink)))
            img = img.resize(rsize, PIL.Image.ANTIALIAS)
            quad /= shrink
            qsize /= shrink

        # Crop.
        border = max(int(np.rint(qsize * 0.1)), 3)
        crop = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))), int(np.ceil(max(quad[:, 0]))),
                int(np.ceil(max(quad[:, 1]))))
        crop = (max(crop[0] - border, 0), max(crop[1] - border, 0), min(crop[2] + border, img.size[0]),
                min(crop[3] + border, img.size[1]))
        if crop[2] - crop[0] < img.size[0] or crop[3] - crop[1] < img.size[1]:
            img = img.crop(crop)
            quad -= crop[0:2]

        # Pad.
        pad = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))), int(np.ceil(max(quad[:, 0]))),
               int(np.ceil(max(quad[:, 1]))))
        pad = (max(-pad[0] + border, 0), max(-pad[1] + border, 0), max(pad[2] - img.size[0] + border, 0),
               max(pad[3] - img.size[1] + border, 0))
        if enable_padding and max(pad) > border - 4:
            pad = np.maximum(pad, int(np.rint(qsize * 0.3)))
            img = np.pad(np.float32(img), ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)), 'reflect')
            h, w, _ = img.shape
            y, x, _ = np.ogrid[:h, :w, :1]
            mask = np.maximum(1.0 - np.minimum(np.float32(x) / pad[0], np.float32(w - 1 - x) / pad[2]),
                              1.0 - np.minimum(np.float32(y) / pad[1], np.float32(h - 1 - y) / pad[3]))
            blur = qsize * 0.02
            img += (scipy.ndimage.gaussian_filter(img, [blur, blur, 0]) - img) * np.clip(mask * 3.0 + 1.0, 0.0, 1.0)
            img += (np.median(img, axis=(0, 1)) - img) * np.clip(mask, 0.0, 1.0)
            img = PIL.Image.fromarray(np.uint8(np.clip(np.rint(img), 0, 255)), 'RGB')
            quad += pad[:2]

        # Transform.
        img = img.transform((transform_size, transform_size), PIL.Image.QUAD, (quad + 0.5).flatten(),
                            PIL.Image.BILINEAR)
        if output_size < transform_size:
            img = img.resize((output_size, output_size), PIL.Image.ANTIALIAS)

        # Save aligned image.
        return img, crop, quad, transform_size

    def rect_to_bb(self, rect):
        # take a bounding predicted by dlib and convert it
        # to the format (x, y, w, h) as we would normally do
        # with OpenCV
        x1 = rect.left()
        y1 = rect.top()
        x2 = rect.right()
        y2 = rect.bottom()

        # return a tuple of (x, y, w, h)
        return (x1, y1), (x2, y2)

    def change_face(self, rgb_face_img):
        try:
            # 		img = PIL.Image.fromarray(np.uint8(np.clip(np.rint(img), 0, 255)), 'RGB')
            if self.if_align_face:
                img, crop, quad, transform_size = self.run_alignment(rgb_face_img)
                I= self.transform(img).unsqueeze(dim=0).to(self.device)
            else:
                transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                ])

                # img = PIL.Image.fromarray(rgb_face_img)
                img = rgb_face_img
                img = transform(img)
                img = img.unsqueeze(dim=0)
                I = F.adaptive_avg_pool2d(img.to(self.device), 256)

            with torch.no_grad():
                self.stylename = list(self.exstyles.keys())[self.style_id]
                self.stylepath = os.path.join(self.DATA_DIR, self.style_type, 'images/train', self.stylename)
                img_rec, instyle = self.encoder(I, randomize_noise=False, return_latents=True,
                                           z_plus_latent=True, return_z_plus_latent=True, resize=False)
                img_rec = torch.clamp(img_rec.detach(), -1, 1)

                latent = torch.tensor(self.exstyles[self.stylename]).repeat(2, 1, 1).to(self.device)
                # latent[0] for both color and structrue transfer and latent[1] for only structrue transfer
                latent[1, 7:18] = instyle[0, 7:18]
                exstyle = self.generator.generator.style(
                    latent.reshape(latent.shape[0] * latent.shape[1], latent.shape[2])).reshape(
                    latent.shape)

                img_gen, _ = self.generator([instyle.repeat(2, 1, 1)], exstyle, z_plus_latent=True,
                                       truncation=0.7, truncation_latent=0, use_res=True,
                                       interp_weights=[0.6] * 7 + [1] * 11)
                img_gen = torch.clamp(img_gen.detach(), -1, 1)
                # deactivate color-related layers by setting w_c = 0
                img_gen2, _ = self.generator([instyle], exstyle[0:1], z_plus_latent=True,
                                        truncation=0.7, truncation_latent=0, use_res=True,
                                        interp_weights=[0.6] * 7 + [0] * 11)
                img_gen2 = torch.clamp(img_gen2.detach(), -1, 1)

            vis = F.adaptive_avg_pool2d(img_gen2, 256)
            img = np.ascontiguousarray(
                ((vis.squeeze(0).cpu().detach().numpy().transpose(1, 2, 0) + 1.0) * 127.5).astype(np.uint8))

            # img is rgb, convert to opencv's default bgr
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            bgr_face_img = np.array(rgb_face_img)
            # Convert RGB to BGR
            bgr_face_img = bgr_face_img[:, :, ::-1].copy()

            # Align faces
            # detector = dlib.get_frontal_face_detector()
            # dets_orig = detector(numpy.array(rgb_face_img)[:, :, ::-1], 1)[0]
            # dets_out = detector(numpy.array(img)[:, :, ::-1], 1)[0]
            #
            # rect_orig_pt1, rect_orig_pt2 = self.rect_to_bb(dets_orig)
            # rect_out_pt1, rect_out_pt2 = self.rect_to_bb(dets_out)
            # bgr_face_img = cv2.rectangle(bgr_face_img, rect_orig_pt1, rect_orig_pt2, (255, 0, 0), 2)
            # img_bgr = cv2.rectangle(img_bgr, rect_out_pt1, rect_out_pt2, (255, 0, 0), 2)

            # cv2.imshow("rgb_face_img", bgr_face_img)
            # cv2.imshow("img", img_bgr)
            # cv2.waitKey(1)
            #
            # fx = (dets_orig.right() - dets_orig.left()) / (dets_out.right() - dets_out.left())
            # fy = (dets_orig.bottom() - dets_orig.top()) / (dets_out.bottom() - dets_out.top())
            #
            # width = int(img.shape[1] * fx)
            # height = int(img.shape[0] * fy)
            #
            # img_bgr_resized = cv2.resize(img_bgr, (width, height), interpolation=cv2.INTER_AREA)
            #
            # # Calculate position in resulting image
            # position_x = dets_orig.left() - int(dets_out.left() * fx)
            # position_y = dets_orig.top() - int(dets_out.top() * fy)
            #
            # crop_x = 0 if position_x >= 0 else abs(position_x)
            # position_x = 0 if position_x < 0 else position_x
            # crop_y = 0 if position_y >= 0 else abs(position_y)
            # position_y = 0 if position_y < 0 else position_y
            #
            # bgr_face_img[position_y:position_y + img_bgr_resized.shape[0] - crop_y, position_x:position_x + img_bgr_resized.shape[1] - crop_x, :] = img_bgr_resized[crop_y:, crop_x:, :]
            #
            # cv2.imshow("plot", img_bgr_resized)
            # cv2.waitKey(1)

            img_bgr_resize = cv2.resize(img_bgr, (crop[2] - crop[0], crop[3] - crop[1]))
            bgr_face_img[crop[1]:crop[3], crop[0]:crop[2]] = img_bgr_resize
            return cv2.cvtColor(bgr_face_img, cv2.COLOR_BGR2RGB)
        except Exception as e:
            print(e)
            return None
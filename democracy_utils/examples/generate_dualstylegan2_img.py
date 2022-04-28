import os

import cv2
import dlib
import numpy as np
import torch
from util import save_image, load_image, visualize
import argparse
from argparse import Namespace
from torchvision import transforms
from torch.nn import functional as F
import torchvision
import matplotlib.pyplot as plt
from model.dualstylegan import DualStyleGAN
from model.sampler.icp import ICPTrainer
from model.encoder.psp import pSp
from PIL import Image
import PIL

MODEL_DIR = 'checkpoint'
DATA_DIR = 'data'
# image_path = './data/content/unsplash-rDEOVtE7vOs.jpg'
image_path = '/home/alejandro/py_workspace/music-utils/democracy_utils/face_changers/dual_style_gan2/data/girl.jpg'
if_align_face = True
device = 'cuda'
style_types = ['cartoon', 'caricature', 'anime', 'arcane', 'comic', 'pixar', 'slamdunk']
style_type = style_types[0]

img_original = cv2.imread(image_path)

if not os.path.exists(os.path.join(MODEL_DIR, style_type)):
    os.makedirs(os.path.join(MODEL_DIR, style_type))

def get_download_model_command(file_id, file_name):
    """ Get wget download command for downloading the desired model and save to directory ../checkpoint/. """
    current_directory = os.getcwd()
    save_path = MODEL_DIR
    url = r"""wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id={FILE_ID}' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id={FILE_ID}" -O {SAVE_PATH}/{FILE_NAME} && rm -rf /tmp/cookies.txt""".format(FILE_ID=file_id, FILE_NAME=file_name, SAVE_PATH=save_path)
    return url

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
download_command = get_download_model_command(file_id=path["id"], file_name=path["name"])
# os.system(download_command)

# download dualstylegan
path = MODEL_PATHS[style_type+'-G']
download_command = get_download_model_command(file_id=path["id"], file_name=os.path.join(style_type, path["name"]))
# os.system(download_command)
# download sampler
path = MODEL_PATHS[style_type+'-N']
download_command = get_download_model_command(file_id=path["id"], file_name=os.path.join(style_type, path["name"]))
# os.system(download_command)
# download extrinsic style code
path = MODEL_PATHS[style_type+'-S']
download_command = get_download_model_command(file_id=path["id"], file_name=os.path.join(style_type, path["name"]))
# os.system(download_command)

transform = transforms.Compose(
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
ckpt = torch.load(os.path.join(MODEL_DIR, style_type, 'generator.pt'), map_location=lambda storage, loc: storage)
generator.load_state_dict(ckpt["g_ema"])
generator = generator.to(device)

# load encoder
model_path = os.path.join(MODEL_DIR, 'encoder.pt')
ckpt = torch.load(model_path, map_location='cpu')
opts = ckpt['opts']
opts['checkpoint_path'] = model_path
opts = Namespace(**opts)
opts.device = device
encoder = pSp(opts)
encoder.eval()
encoder = encoder.to(device)

# load extrinsic style code
exstyles = np.load(os.path.join(MODEL_DIR, style_type, MODEL_PATHS[style_type+'-S']["name"]), allow_pickle='TRUE').item()

# load sampler network
icptc = ICPTrainer(np.empty([0,512*11]), 128)
icpts = ICPTrainer(np.empty([0,512*7]), 128)
ckpt = torch.load(os.path.join(MODEL_DIR, style_type, 'sampler.pt'), map_location=lambda storage, loc: storage)
icptc.icp.netT.load_state_dict(ckpt['color'])
icpts.icp.netT.load_state_dict(ckpt['structure'])
icptc.icp.netT = icptc.icp.netT.to(device)
icpts.icp.netT = icpts.icp.netT.to(device)

print('Model successfully loaded!')

original_image = load_image(image_path)

def run_alignment(image_path):
    import dlib
    from model.encoder.align_all_parallel import align_face
    modelname = os.path.join(MODEL_DIR, 'shape_predictor_68_face_landmarks.dat')
    if not os.path.exists(modelname):
        import wget, bz2
        wget.download('http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2', modelname+'.bz2')
        zipfile = bz2.BZ2File(modelname+'.bz2')
        data = zipfile.read()
        open(modelname, 'wb').write(data)
    predictor = dlib.shape_predictor(modelname)
    aligned_image, crop, quad, input_size = align_face(filepath=image_path, predictor=predictor)
    return aligned_image, crop, quad, input_size

if if_align_face:
    aligned_image, crop, quad, input_size = run_alignment(image_path)
    I = transform(aligned_image).unsqueeze(dim=0).to(device)
else:
    I = F.adaptive_avg_pool2d(load_image(image_path).to(device), 256)

style_id = 26

# try to load the style image
stylename = list(exstyles.keys())[style_id]
stylepath = os.path.join(DATA_DIR, style_type, 'images/train', stylename)
print('loading %s'%stylepath)
if os.path.exists(stylepath):
    S = load_image(stylepath)
    fig = plt.figure(figsize=(10,10),dpi=30)
    visualize(S[0])
    fig.canvas.draw()

    # convert canvas to image
    img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    # img is rgb, convert to opencv's default bgr
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # display image with opencv or any operation you like
    cv2.imshow("plot", img)
    k = cv2.waitKey(0)

else:
    print('%s is not found'%stylename)

with torch.no_grad():
    img_rec, instyle = encoder(I, randomize_noise=False, return_latents=True,
                               z_plus_latent=True, return_z_plus_latent=True, resize=False)
    img_rec = torch.clamp(img_rec.detach(), -1, 1)

    latent = torch.tensor(exstyles[stylename]).repeat(2, 1, 1).to(device)
    # latent[0] for both color and structrue transfer and latent[1] for only structrue transfer
    latent[1, 7:18] = instyle[0, 7:18]
    exstyle = generator.generator.style(latent.reshape(latent.shape[0] * latent.shape[1], latent.shape[2])).reshape(
        latent.shape)

    img_gen, _ = generator([instyle.repeat(2, 1, 1)], exstyle, z_plus_latent=True,
                           truncation=0.7, truncation_latent=0, use_res=True, interp_weights=[0.6] * 7 + [1] * 11)
    img_gen = torch.clamp(img_gen.detach(), -1, 1)
    # deactivate color-related layers by setting w_c = 0
    img_gen2, _ = generator([instyle], exstyle[0:1], z_plus_latent=True,
                            truncation=0.7, truncation_latent=0, use_res=True, interp_weights=[0.6] * 7 + [0] * 11)
    img_gen2 = torch.clamp(img_gen2.detach(), -1, 1)

# vis = torchvision.utils.make_grid(F.adaptive_avg_pool2d(torch.cat([img_rec, img_gen, img_gen2], dim=0), 256), 4, 1)
# fig = plt.figure(figsize=(10,10),dpi=120)
# visualize(vis.cpu())
# fig.canvas.draw()

# convert canvas to image
vis = F.adaptive_avg_pool2d(img_gen2, 256)
img = np.ascontiguousarray(((vis.squeeze(0).cpu().detach().numpy().transpose(1, 2, 0) + 1.0) * 127.5).astype(np.uint8))
# img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))

# img is rgb, convert to opencv's default bgr
img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

img_original = cv2.imread(image_path)
# display image with opencv or any operation you like
cv2.namedWindow("result", cv2.WINDOW_NORMAL)  # Create window with freedom of dimensions
cv2.imshow("result", img)
cv2.namedWindow("original", cv2.WINDOW_NORMAL)  # Create window with freedom of dimensions
cv2.imshow("original", img_original)

img_resize = cv2.resize(img, (crop[2] - crop[0], crop[3] - crop[1]))
# img_resize = img.copy()
h, w, c = img_resize.shape

# Inverse quad
# im_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
# inverse_quad = quad.copy()
# temp_quad = quad.copy()
#
# temp_quad[0][0] = w - quad[0][0]
# temp_quad[1][0] = w - quad[1][0]
# temp_quad[2][0] = w - quad[2][0]
# temp_quad[3][0] = w - quad[3][0]
#
# temp_quad[0][1] = h - quad[0][1]
# temp_quad[1][1] = h - quad[1][1]
# temp_quad[2][1] = h - quad[2][1]
# temp_quad[3][1] = h - quad[3][1]
#
# inverse_quad[0] = temp_quad[3]
# inverse_quad[1] = temp_quad[2]
# inverse_quad[2] = temp_quad[1]
# inverse_quad[3] = temp_quad[0]
#
# im_pil = im_pil.transform((crop[2] - crop[0], crop[3] - crop[1]), PIL.Image.QUAD, (inverse_quad + 0.5).flatten(), PIL.Image.BILINEAR)
# img_inverse_quad = cv2.cvtColor(np.array(im_pil), cv2.COLOR_RGB2BGR)

img_original_resize = cv2.resize(img_original, (256, 256))


from model.face_aligner import FaceAligner
# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor and the face aligner
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('/home/alejandro/fresssh/py_workspace/DualStyleGAN/shape_predictor_68_face_landmarks.dat')
fa = FaceAligner(predictor, desiredFaceWidth=256)

# load the input image, resize it, and convert it to grayscale

gray = cv2.cvtColor(img_original_resize, cv2.COLOR_BGR2GRAY)
# show the original input image and detect faces in the grayscale
# image
rects = detector(gray, 2)

faceAligned, m_affine, (w_affine_orig, h_affine_orig) = fa.align(img_original_resize, gray, rects[0])
cv2.namedWindow("orig_aligned", cv2.WINDOW_NORMAL)  # Create window with freedom of dimensions
cv2.imshow("orig_aligned", faceAligned)

m_affine_inverse = cv2.invertAffineTransform(m_affine).copy()

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# show the original input image and detect faces in the grayscale
# image
rects = detector(gray, 2)

faceAligned, m_affine, (w_affine, h_affine) = fa.align(img, gray, rects[0])
cv2.namedWindow("out_aligned", cv2.WINDOW_NORMAL)  # Create window with freedom of dimensions
cv2.imshow("out_aligned", faceAligned)

output_realigned = cv2.warpAffine(img, m_affine, (w_affine, h_affine),
                        flags=cv2.INTER_CUBIC)
output_realigned = cv2.warpAffine(output_realigned, m_affine_inverse, (w_affine, h_affine),
                        flags=cv2.INTER_CUBIC)

cv2.namedWindow("re_out_aligned", cv2.WINDOW_NORMAL)  # Create window with freedom of dimensions
cv2.imshow("re_out_aligned", output_realigned)




img_output = img_original.copy()
img_output[crop[1]:crop[3], crop[0]:crop[2]] = img_resize
cv2.namedWindow("output", cv2.WINDOW_NORMAL)  # Create window with freedom of dimensions
cv2.imshow("output", img_output)
k = cv2.waitKey(0)


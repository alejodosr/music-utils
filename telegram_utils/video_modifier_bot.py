import glob
import random

from telegram.ext.updater import Updater
from telegram.update import Update
from telegram.ext.callbackcontext import CallbackContext
from telegram.ext.commandhandler import CommandHandler
from telegram.ext.messagehandler import MessageHandler
from telegram.ext.filters import Filters
import os
from video_utils.generate_video_democracy import generate_video_style
from video_utils.backgroundremover.backgroundremover.cmd.api import process_video
from youtube_utils.search_and_download_yt import download_from_youtube
from text_to_img_utils.prompt_stable_difussion import get_samples_from_stable_diffusion


BOT_API_KEY = os.getenv('BOT_API_KEY', default=None)
VIDEO_TYPE = 'fresssh'
VIDEO_SUBTYPE = '0'
BLOCKED_EXECUTION = False
AVAILABLE_COMMANDS = ['anime', 'dual', 'fresssh', 'green', 'youtube', 'diffusion']

updater = Updater(BOT_API_KEY,
                  use_context=True,
                  base_url='http://0.0.0.0:8081/bot')

def start(update: Update, context: CallbackContext):
    update.message.reply_text(
        "Hi, which video you want to convert?")

def unknown(update: Update, context: CallbackContext):
    update.message.reply_text(
        f"Sorry '{update.message.text}' is not a valid command ({AVAILABLE_COMMANDS})")

def video_type(update: Update, context: CallbackContext):
    update.message.reply_text(
        "El video te lo doy pronto")

def fresssh_type(update: Update, context: CallbackContext):
    global VIDEO_TYPE
    VIDEO_TYPE = 'fresssh'

def anime_type(update: Update, context: CallbackContext):
    global VIDEO_TYPE
    VIDEO_TYPE = 'anime'

def green_type(update: Update, context: CallbackContext):
    global VIDEO_TYPE
    VIDEO_TYPE = 'green'

def youtube_type(update: Update, context: CallbackContext):
    update.message.reply_text(f"Downloading video: {context.args[0]}")
    video_output = download_from_youtube(context.args[0])
    update.message.reply_document(open(video_output, 'rb'), filename='youtube.mp4')

def diffusion_type(update: Update, context: CallbackContext):


    if len(context.args):
        idcs = []
        for idx, arg in enumerate(context.args):
            if '/' in arg:
                idcs.append(idx)
                if len(idcs) == 2:
                    break

        prompt = ' '.join(context.args[idcs[0]:idcs[1]+1]).replace('/', '')
        seed = random.randint(0, 1e4)
        outdir = '/tmp/sd_queries'
        config_root = '/home/alejandro/py_workspace/stable-diffusion'
        config = os.path.join(config_root, 'configs/stable-diffusion/v1-inference.yaml')
        model = os.path.join(config_root, 'models/ldm/stable-diffusion-v1/model.ckpt')

        update.message.reply_text(f"Querying Stable Diffusion with: '{prompt}' and seed ({seed})")
        get_samples_from_stable_diffusion(prompt, seed=seed, outdir=outdir, config=config, ckpt=model)
        for idx, filename in enumerate(glob.glob(f'{outdir}/samples/*.png')):
            update.message.reply_document(open(filename, 'rb'), filename=f'sample{idx}.png')
        os.system(f'rm -r {outdir}')
    else:
        update.message.reply_text(f"Not a valid call for diffusion")

def dual0_type(update: Update, context: CallbackContext):
    global VIDEO_TYPE, VIDEO_SUBTYPE
    VIDEO_TYPE = 'dual'
    VIDEO_SUBTYPE = '0'

def downloader(update, context):
    context.bot.get_file(update.message.document).download()

    update.message.reply_text(
        "Tengo el video maquina")
    # writing to a custom file
    # with open("custom/file.doc", 'wb') as f:
    #     context.bot.get_file(update.message.document).download(out=f)

def downloader_video(update, context):
    global BLOCKED_EXECUTION
    if not BLOCKED_EXECUTION:
        BLOCKED_EXECUTION = True

        if os.path.isdir('video'):
            os.system(f'rm -rf ./video')

        update.message.reply_text(
            f"Hello Fresssh \U0001F343, good to see you. I'm downloading the video \U0001F619")

        # writing to a custom file
        if not os.path.isdir('video'):
            os.system('mkdir video')
        try:
            with open("video/video.mp4", 'wb') as f:
                context.bot.get_file(update.message.video).download(out=f)
        except:
            with open("video/video.mp4", 'wb') as f:
                context.bot.get_file(update.message.document).download(out=f)

        update.message.reply_text(
            f"Processing video with '{VIDEO_TYPE}' style...")

        if VIDEO_TYPE == 'green':
            video_output = process_video('video/video.mp4', type='greenvideo')
        elif VIDEO_TYPE in AVAILABLE_COMMANDS:
            video_output = generate_video_style('video/video.mp4', VIDEO_TYPE, VIDEO_SUBTYPE)
        else:
            update.message.reply_text(
                f"Sorry ma friend, only available commands are {AVAILABLE_COMMANDS}")


        print(video_output)

        update.message.reply_text(
            f"Here you are the video \U0001F308, rock it baby!")

        update.message.reply_document(open(video_output, 'rb'), filename=video_output.split(os.pathsep)[-1])

        BLOCKED_EXECUTION = False
    else:
        update.message.reply_text(
            f"Sorry, there's an ongoing work here, wait til it finishes")


updater.dispatcher.add_handler(CommandHandler('anime', anime_type))
updater.dispatcher.add_handler(CommandHandler('dual0', dual0_type))
updater.dispatcher.add_handler(CommandHandler('fresssh', fresssh_type))
updater.dispatcher.add_handler(CommandHandler('green', green_type))
updater.dispatcher.add_handler(CommandHandler('youtube', youtube_type, pass_args=True))
updater.dispatcher.add_handler(CommandHandler('diffusion', diffusion_type, pass_args=True))
updater.dispatcher.add_handler(MessageHandler(Filters.document, downloader_video))
updater.dispatcher.add_handler(MessageHandler(Filters.video, downloader_video))

# updater.dispatcher.add_handler(MessageHandler(Filters.text, video_type))
updater.dispatcher.add_handler(MessageHandler(
    # Filters out unknown commands
    Filters.command, unknown))

updater.start_polling()

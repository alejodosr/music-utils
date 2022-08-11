from telegram.ext.updater import Updater
from telegram.update import Update
from telegram.ext.callbackcontext import CallbackContext
from telegram.ext.commandhandler import CommandHandler
from telegram.ext.messagehandler import MessageHandler
from telegram.ext.filters import Filters
import os
from video_utils.generate_video_democracy import generate_video_style
from video_utils.backgroundremover.backgroundremover.cmd.api import process_video

BOT_API_KEY = os.getenv('BOT_API_KEY', default=None)
VIDEO_TYPE = 'fresssh'
VIDEO_SUBTYPE = '0'
BLOCKED_EXECUTION = False

updater = Updater(BOT_API_KEY,
                  use_context=True)

def start(update: Update, context: CallbackContext):
    update.message.reply_text(
        "Hi, which video you want to convert?")

def unknown(update: Update, context: CallbackContext):
    update.message.reply_text(
        "Sorry '%s' is not a valid command" % update.message.text)

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
        with open("video/video.mp4", 'wb') as f:
            context.bot.get_file(update.message.video).download(out=f)

        update.message.reply_text(
            f"Processing video with '{VIDEO_TYPE}' style...")

        if VIDEO_TYPE != 'green':
            video_output = generate_video_style('video/video.mp4', VIDEO_TYPE, VIDEO_SUBTYPE)
        else:
            video_output = process_video('video/video.mp4', type='greenvideo')
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
updater.dispatcher.add_handler(MessageHandler(Filters.document, downloader))
updater.dispatcher.add_handler(MessageHandler(Filters.video, downloader_video))

# updater.dispatcher.add_handler(MessageHandler(Filters.text, video_type))
updater.dispatcher.add_handler(MessageHandler(
    # Filters out unknown commands
    Filters.command, unknown))

updater.start_polling()
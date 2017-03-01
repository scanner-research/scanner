import os.path

try:
    import youtube_dl
except ImportError:
    print('You need to install youtube-dl to run this. Try running:\npip install youtube-dl')
    exit()

VID_PATH = "/tmp/example.mp4"

def download_video():
    if not os.path.isfile(VID_PATH):
        ydl_opts = {
            'format': 'mp4',
            'outtmpl': u'/tmp/example.%(ext)s'
        }
        with youtube_dl.YoutubeDL(ydl_opts) as ydl:
            ydl.download(["https://www.youtube.com/watch?v=dQw4w9WgXcQ"])
    return VID_PATH

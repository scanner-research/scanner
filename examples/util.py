import os.path

try:
    import requests
except ImportError:
    print(
        'You need to install requests to run this. Try running:\npip install requests'
    )
    exit()

VID_URL = "https://storage.googleapis.com/scanner-data/public/sample-clip.mp4"
VID_PATH = '/tmp/example.mp4'


def download_video():
    if not os.path.isfile(VID_PATH):
        with open(VID_PATH, 'wb') as f:
            resp = requests.get(VID_URL, stream=True)
            assert resp.ok
            for block in resp.iter_content(1024):
                f.write(block)
            f.flush()
    return VID_PATH

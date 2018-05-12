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

IMG_PATH = '/tmp/example.mp4'


def download_video():
    if not os.path.isfile(VID_PATH):
        with open(VID_PATH, 'wb') as f:
            resp = requests.get(VID_URL, stream=True)
            assert resp.ok
            for block in resp.iter_content(1024):
                f.write(block)
            f.flush()
    return VID_PATH


def download_images():
    img_template = (
        'https://storage.googleapis.com/scanner-data/public/sample-frame-{:d}.jpg')
    output_template = 'sample-frame-{:d}.jpg'

    for i in range(1, 4):
        with open(output_template.format(i), 'wb') as f:
            resp = requests.get(img_template.format(i), stream=True)
            assert resp.ok
            for block in resp.iter_content(1024):
                f.write(block)
            f.flush()

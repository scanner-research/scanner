from .config import mkdir_p
import os
import urllib.request, urllib.error, urllib.parse
import errno
import tarfile


def temp_directory():
    path = os.path.expanduser('~/.scanner/resources')
    mkdir_p(path)
    return path


def download_temp_file(url, local_path=None, untar=False):
    if local_path is None:
        local_path = url.rsplit('/', 1)[-1]
    local_path = os.path.join(temp_directory(), local_path)
    mkdir_p(os.path.dirname(local_path))
    if not os.path.isfile(local_path):
        print('Downloading {:s} to {:s}...'.format(url, local_path))
        f = urllib.request.urlopen(url)
        with open(local_path, 'wb') as local_f:
            local_f.write(f.read())

        if untar:
            with tarfile.open(local_path) as tar_f:
                tar_f.extractall(temp_directory())
    if untar:
        return temp_directory()
    else:
        return local_path


def default(d, k, v):
    if k not in d:
        return v() if callable(v) else v
    return d[k]


VID_URL = "https://storage.googleapis.com/scanner-data/public/sample-clip.mp4"
VID_PATH = '/tmp/example.mp4'

IMG_PATH = '/tmp/example.mp4'


def download_video():
    try:
        import requests
    except ImportError:
        print(
            'You need to install requests to run this. Try running:\npip3 install requests'
        )
        exit()

    if not os.path.isfile(VID_PATH):
        with open(VID_PATH, 'wb') as f:
            resp = requests.get(VID_URL, stream=True)
            assert resp.ok
            for block in resp.iter_content(1024):
                f.write(block)
            f.flush()
    return VID_PATH


def download_images():
    try:
        import requests
    except ImportError:
        print(
            'You need to install requests to run this. Try running:\npip3 install requests'
        )
        exit()

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

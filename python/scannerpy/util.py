from .config import mkdir_p
import os
import urllib.request, urllib.error, urllib.parse
import errno
import tarfile
from contextlib import contextmanager
import tempfile


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


@contextmanager
def sample_video(delete=True):
    try:
        import requests
    except ImportError:
        print(
            'You need to install requests to run this. Try running:\npip3 install requests'
        )
        exit()

    url = "https://storage.googleapis.com/scanner-data/public/sample-clip.mp4"

    if delete:
        f = tempfile.NamedTemporaryFile(suffix='.mp4')
    else:
        sample_path = '/tmp/sample_video.mp4'
        if os.path.isfile(sample_path):
            yield sample_path
            return

        f = open(sample_path, 'wb')

    with f as f:
        resp = requests.get(url, stream=True)
        assert resp.ok
        for block in resp.iter_content(1024):
            f.write(block)
        f.flush()
        yield f.name

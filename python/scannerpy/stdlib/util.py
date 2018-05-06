from ..config import mkdir_p
import os
import urllib.request, urllib.error, urllib.parse
import errno


def temp_directory():
    path = os.path.expanduser('~/.scanner/resources')
    mkdir_p(path)
    return path


def download_temp_file(url, local_path=None):
    if local_path is None:
        local_path = url.rsplit('/', 1)[-1]
    local_path = os.path.join(temp_directory(), local_path)
    mkdir_p(os.path.dirname(local_path))
    if not os.path.isfile(local_path):
        print('Downloading {:s} to {:s}...'.format(url, local_path))
        f = urllib.request.urlopen(url)
        with open(local_path, 'wb') as local_f:
            local_f.write(f.read())
    return local_path


def default(d, k, v):
    if k not in d:
        return v() if callable(v) else v
    return d[k]

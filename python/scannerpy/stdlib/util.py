from __future__ import absolute_import, division, print_function, unicode_literals
import os
import urllib2
import errno


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


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
        f = urllib2.urlopen(url)
        with open(local_path, 'wb') as local_f:
            local_f.write(f.read())
    return local_path


def default(d, k, v):
    if k not in d:
        return v() if callable(v) else v
    return d[k]


# def download_temp_youtube_video(url, local_path=None):
#     # Get filename
#     https://youtu.be/SHoHdkUw-Is  '--restrict-filenames --get-filename'
#     ~/.local/bin/youtube-dl -f 266 https://youtu.be/SHoHdkUw-Is -o
#     if local_path is None:
#         local_path = url.rsplit('/', 1)[-1]
#     local_path = os.path.join(temp_directory(), local_path)
#     mkdir_p(os.path.dirname(local_path))
#     if not os.path.isfile(local_path):
#         print('Downloading {:s} to {:s}...'.format(url, local_path))
#         f = urllib2.urlopen(url)
#         with open(local_path, 'wb') as local_f:
#             local_f.write(f.read())
#     return local_path

# ~/.local/bin/youtube-dl -f 266 https://youtu.be/SHoHdkUw-Is -o

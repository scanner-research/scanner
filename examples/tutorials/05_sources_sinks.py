from scannertools.storage.files import FilesStream
from typing import Sequence
import scannertools.imgproc
import scannerpy as sp
import cv2

import sys
import os.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/..')
import util

################################################################################
# This tutorial shows how to use Sources and Sinks for reading data from       #
# places other than tables and columns.                                        #
################################################################################

def main():
    cl = sp.Client()

    # What if, instead of a video, you had a list of image files that you
    # wanted to process? Scanner provides an extensible interface for reading and
    # writing data to locations other than the database.

    # For example, let's download a few images now and create a list of their paths:

    util.download_images()
    image_paths = ['sample-frame-1.jpg', 'sample-frame-2.jpg', 'sample-frame-3.jpg']

    # Scanner provides a built-in source to read files from the local filesystem:

    image_stream = FilesStream(image_paths)

    compressed_images = cl.io.Input([image_stream])
    # Like with sc.sources.FrameColumn, we will bind the inputs to this source when
    # we define a job later on.

    # Let's write a pipeline that reads our images, resizes them, and writes them
    # back out as files to the filesystem.

    # Since the input images are compressed, we decompress them with the
    # ImageDecoder
    frames = cl.ops.ImageDecoder(img=compressed_images)

    resized_frames = cl.ops.Resize(frame=frames, width=[640], height=[360])

    # Rencode the image to jpg
    encoded_frames = cl.ops.ImageEncoder(frame=resized_frames, format='jpg')

    # Write the compressed images to files
    resized_paths = ['resized-1.jpg', 'resized-2.jpg', 'resized-3.jpg']
    resized_stream = FilesStream(resized_paths)
    output = cl.io.Output(encoded_frames, [resized_stream])

    cl.run(output, sp.PerfParams.estimate(), cache_mode=sp.CacheMode.Overwrite)

    print('Finished! Wrote the following images: ' + ', '.join(resized_paths))

    # If you want to learn how write your own custom Source or Sink in C++, check
    # out tutorials 09_defining_cpp_sources.py and 10_defining_cpp_sinks.py

if __name__ == "__main__":
    main()

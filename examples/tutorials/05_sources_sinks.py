import scannerpy
from scannerpy import Database, Job, DeviceType
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
    db = Database()

    # What if, instead of a video, you had a list of image files that you
    # wanted to process? Scanner provides an extensible interface for reading and
    # writing data to locations other than the database.

    # For example, let's download a few images now and create a list of their paths:

    util.download_images()
    image_paths = ['sample-frame-1.jpg', 'sample-frame-2.jpg', 'sample-frame-3.jpg']

    # Scanner provides a built-in source to read files from the local filesystem:

    compressed_images = db.sources.Files()
    # Like with db.sources.FrameColumn, we will bind the inputs to this source when
    # we define a job later on.

    # Let's write a pipeline that reads our images, resizes them, and writes them
    # back out as files to the filesystem.

    # Since the input images are compressed, we decompress them with the
    # ImageDecoder
    frame = db.ops.ImageDecoder(img=compressed_images)

    resized = db.ops.Resize(frame=frame, width=640, height=360)

    # Rencode the image to jpg
    encoded_frame = db.ops.ImageEncoder(frame=resized, format='jpg')

    # Write the compressed images to files
    output = db.sinks.Files(input=encoded_frame)

    resized_paths = ['resized-1.jpg', 'resized-2.jpg', 'resized-3.jpg']

    job = Job(op_args={
        compressed_images: {'paths': image_paths},
        output: {'paths': resized_paths}
    })

    db.run(output=output, jobs=[job])

    print('Finished! Wrote the following images: ' + ', '.join(resized_paths))

    # If you want to learn how write your own custom Source or Sink in C++, check
    # out tutorials 09_defining_cpp_sources.py and 10_defining_cpp_sinks.py

if __name__ == "__main__":
    main()

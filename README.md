# Scanner: Efficient Video Analysis at Scale [![Build Status](https://travis-ci.org/scanner-research/scanner.svg?branch=master)](https://travis-ci.org/scanner-research/scanner) #

_To try out Scanner, [check out our Quick start](https://github.com/scanner-research/scanner#quick-start)._

Scanner is like Spark for videos. It runs stateful functions across video frames using clusters of machines with CPUs and GPUs. For example, you could use Scanner to:

* [Locate and recognize faces](https://github.com/scanner-research/scanner/blob/master/examples/face_detection/face_detect.py)
* [Detect shots in a film](https://github.com/scanner-research/scanner/blob/master/examples/shot_detection/shot_detect.py)
* [Search videos by image](https://github.com/scanner-research/scanner/blob/master/examples/reverse_image_search/search.py)

[Click here to learn more about the design and usage of Scanner.](https://github.com/scanner-research/scanner/wiki/Getting-started)

Scanner provides a Python API to organize your videos and run high-performance functions written in C++. For example, this program computes a histogram of colors for each frame in a set of videos on the GPU:

```python
from scannerpy import Database, DeviceType
from scannerpy.stdlib import parsers
db = Database()
videos = db.ingest_video_collection('my_videos', ['vid0.mp4', 'vid1.mkv'])
hist = db.ops.Histogram(device=DeviceType.GPU)
output = db.run(videos, hist, 'my_videos_hist')
vid0_hists = output.tables(0).columns(0).load(parsers.histograms)
```

[Click here to see more code examples of using Scanner.](https://github.com/scanner-research/scanner/tree/master/examples/tutorial)

Scanner makes it easy to use existing computer vision and pixel processing tools. For example, Scanner has:

* [Caffe](https://github.com/bvlc/caffe) support for neural network evaluation
* [OpenCV](https://github.com/opencv/opencv) support with included kernels for color histograms and optical flow
* Object tracking in videos with [Struck](https://github.com/samhare/struck)
* Image processing with [Halide](http://halide-lang.org/)

Scanner is an active research project, part of a collaboration between Carnegie Mellon and Stanford. Please contact [Alex Poms](https://github.com/apoms) and [Will Crichton](https://github.com/willcrichton) with questions.

## Quick start ##

To quickly dive into Scanner, you can use one of our prebuilt [Docker images](https://hub.docker.com/r/scannerresearch/scanner). To run a GPU image, you must install and use [nvidia-docker](https://github.com/NVIDIA/nvidia-docker).

```bash
nvidia-docker run -d --name scanner -ti scannerresearch/scanner:gpu /bin/bash
nvidia-docker attach scanner
```

_Note: if you don't have a GPU, then run `docker` instead of `nvidia-docker` and use `scanner:cpu` instead of `scanner:gpu` in the Docker image name._

Then inside your Docker container, run:

```bash
python examples/face_detection/face_detect.py
```

This runs a Scanner demo which detects faces in every frame of a short video from YouTube, creating a file `example_faces.mp4`. Type `Ctrl-P + Ctrl-Q` to detach from the container and then run:

```bash
nvidia-docker cp scanner:/opt/scanner/example_faces.mp4 .
```

Then you can view the generated video on your own machine. That's it! To learn more about Scanner, please visit the [Scanner wiki](https://github.com/scanner-research/scanner/wiki).

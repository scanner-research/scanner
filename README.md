# Scanner: Efficient Video Analysis at Scale [![Build Status](https://travis-ci.org/scanner-research/scanner.svg?branch=master)](https://travis-ci.com/apoms/scanner) #

_For [build instructions](TODO), [tutorials](TODO), [documentation](TODO), and [contributing guidelines](https://github.com/apoms/scanner/wiki/Contributing), visit the [Scanner wiki](https://github.com/apoms/scanner/wiki)._

Scanner lets you write stateful functions that get efficiently mapped across batches of video frames. These functions can execute on a multi-core CPU or GPU and can be distributed across multiple machines. You can think about Scanner like Spark for pixels. For example, you could use Scanner to:

* Example 1
* Example 2
* Example 3

To support these applications, Scanner uses a Python interface similar to Tensorflow and Spark SQL. Videos are represented as tables in a database, and users write computation graphs to transform these tables. For example, to compute the color histogram for each frame in a set of videos on the GPU:

```python
from scannerpy import Database, DeviceType
from scannerpy.stdlib import parsers
db = Database()
videos = db.ingest_video_collection('my_videos', ['vid1.mp4', 'vid2.mkv'])
hist = db.ops.Histogram(device=DeviceType.GPU)
output = db.run(videos, hist, 'my_videos_hist')
vid1_hists = output.tables(0).columns(0).load(parsers.histograms)
```

Scanner provides a convenient way to organize your videos as well as data derived from the videos (bounding boxes, histograms, feature maps, etc.) using a relational database. Behind the scenes, Scanner handles decoding the compressed videos into raw frames, allowing you to process an individual video in parallel. It then runs a computation graph on the decoded frames using kernels written in C++ for maximum performance and distributes the computation over a cluster. Scanner supports a number of operators and third-party libraries to reduce the work of writing new computations:

* [Caffe](https://github.com/bvlc/caffe) support for neural network evaluation
* [OpenCV](https://github.com/opencv/opencv) support with included kernels for color histograms and optical flow
* Object tracking in videos with [Struck](https://github.com/samhare/struck)
* Image processing with [Halide](http://halide-lang.org/)

Lastly, Scanner also offers some utilities for ease of development:

* Profiling via [chrome://tracing](https://www.chromium.org/developers/how-tos/trace-event-profiling-tool)
* Support for different storage backends including [Google Cloud Storage](https://cloud.google.com/storage/)
* Custom operators for adding your own functionality outside the source tree

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
python examples/demo/demo.py
```

This runs a Scanner demo which detects faces in every frame of a short video from YouTube, creating a file `example_faces.mp4`. Type `Ctrl-P + Ctrl-Q` to detach from the container and then run:

```bash
nvidia-docker cp scanner:/opt/scanner/example_faces.mp4 .
```

Then you can view the generated video on your own machine. That's it! To learn more about Scanner, please visit the [Scanner wiki](https://github.com/apoms/scanner/wiki).

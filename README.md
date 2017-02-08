# Scanner #
[![Build Status](https://travis-ci.com/apoms/scanner.svg?token=3riCqXaXCxyYqpsVk2yv&branch=master)](https://travis-ci.com/apoms/scanner)

_For [build instructions](TODO), [tutorials](TODO), [documentation](TODO), and [contributing guidelines](TODO), visit the [Scanner wiki](https://github.com/apoms/scanner/wiki)._

Scanner is a system for efficient analysis of videos at scale. It lets you write stateful functions that get efficiently mapped across batches of video frames. These functions can execute on a multi-core CPU or GPU and can be distributed across multiple machines. You can think about Scanner like Spark for pixels. For example, you could use Scanner to:

* Example 1
* Example
* Example 3

To do these kinds of applications, Scanner exposes a Python interface similar to Tensorflow and Spark SQL. Videos are represented as tables in a database, and users write computation graphs to transform these tables. For example, to compute the color histogram for each frame in a set of videos on the GPU:

```python
from scannerpy import Database, DeviceType
db = Database()
input = db.ingest_video_collection('my_videos', ['vid1.mp4', 'vid2.mkv'])
hist = db.ops.Histogram(device=DeviceType.GPU)
output = db.run(input, hist, 'my_videos_hist')
vid1_hists = output.tables(0).columns(0).load()
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
nvidia-docker run --name scanner -ti scannerresearch/scanner:ubuntu16.04-cuda8.0-cv3.1.0 /bin/bash
```

This Docker container comes prebuilt with the [k-nearest neighbors](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm) pipeline in `scanner/pipelines/knn_pipeline.cpp`. It does reverse image search (like with Google Images) by computing deep features for each frame of the input video, and then comparing the features from query image against the video's. Here, Scanner pre-computes the video features (`scanner_server run example_job example_dataset`), and then a Python script `knn.py` interactively queries against those features with a standard KNN implementation.

Try running this inside your Docker container:

```bash
# Get a video to test on
youtube-dl -f mp4 -o "example.%(ext)s" "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
echo "example.mp4" > videos.txt

# Extract features from the video
./build/scanner_server ingest example_dataset videos.txt
./build/scanner_server run example_dataset base knn example_job

# Compute k-nearest neighbors on an exemplar
wget -O query.jpg https://upload.wikimedia.org/wikipedia/en/9/9b/Rickastleyposter.jpg
python python/knn.py example_dataset example_job
# Give "query.jpg" as the input to the prompt, then replace FRAMENUMBER below with one of the frame numbers
ffmpeg -i example.mp4 -vf "select=eq(n\,FRAMENUMBER)" -vframes 1 result.png
```

From outside the the container, run `nvidia-docker cp scanner:/opt/scanner/result.png .` to get the query result. That's it!

To learn more about Scanner, please visit the [Scanner wiki](https://github.com/apoms/scanner/wiki).

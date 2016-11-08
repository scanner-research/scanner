# Scanner #
[![Build Status](https://travis-ci.com/apoms/scanner.svg?token=3riCqXaXCxyYqpsVk2yv&branch=master)](https://travis-ci.com/apoms/scanner)

_For build instructions, tutorials, documentation, and contributing guidelines, visit the [Scanner wiki](https://github.com/apoms/scanner/wiki)._

Scanner is a system for low-level, high-performance batch processing of images and videos, or visual data. It lets you write functions that get efficiently mapped across batches of frames. These functions can execute on a multi-core CPU or GPU and can be distributed across multiple machines. You can think about it like Hadoop for pixels. For example, you could use Scanner to make an application to:

* Compute screen time for each actor in a movie database
* Classify cell phenotypes from microscope imaging
* Blur all the faces in a set of video
* Implement reverse image search over a large image collection

To write these applications, a user provides Scanner a pipeline of operations. For example, to blur faces in a video, the pipeline is:

1. Decode video into frames
2. Find faces in each frame
3. Blur each face in each frame
4. Encode the blurred frames into a new video

Scanner then takes this pipeline and runs it in parallel over batches of frames. Scanner also provides infrastructure for performing I/O, i.e. loading the input images/videos from disk and saving the results back to disk.

Additionally, Scanner includes a library of common pipeline components for reducing the work of writing new pipelines:

* Parallel video decode/encode on software or dedicated hardware
* [Caffe](https://github.com/bvlc/caffe) support for neural network evaluation
* [OpenCV](https://github.com/opencv/opencv) support with included kernels for color histograms and optical flow
* Object tracking in videos with [Struck](https://github.com/samhare/struck)

Lastly, Scanner also offers several utilities for ease of development:

* Profiling via [chrome://tracing](https://www.chromium.org/developers/how-tos/trace-event-profiling-tool)
* Support for different storage backends including [Google Cloud Storage](https://cloud.google.com/storage/)
* Python interface for extracting results from the database
* Server and web interface for visualizing Scanner query results

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

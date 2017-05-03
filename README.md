# Scanner: Efficient Video Analysis at Scale [![Build Status](https://travis-ci.org/scanner-research/scanner.svg?branch=master)](https://travis-ci.org/scanner-research/scanner) #

_To try out Scanner, [check out our Quick start](https://github.com/scanner-research/scanner#quick-start) and [browse our Wiki](https://github.com/scanner-research/scanner/wiki)._

Scanner is like Spark for videos. It runs stateful functions across video frames using clusters of machines with CPUs and GPUs. For example, you could use Scanner to:

* [Locate and recognize faces in a video](https://github.com/scanner-research/scanner/blob/master/examples/face_detection/face_detect.py)
* [Detect shots in a film](https://github.com/scanner-research/scanner/blob/master/examples/shot_detection/shot_detect.py)
* [Search videos by image](https://github.com/scanner-research/scanner/blob/master/examples/reverse_image_search/search.py)

[Click here to learn more about the design and usage of Scanner.](https://github.com/scanner-research/scanner/wiki/Getting-started)

Scanner provides a Python API to organize your videos and run high-performance functions written in C++. For example, this program computes a histogram of colors for each frame in a set of videos on the GPU:

```python
from scannerpy import Database, DeviceType, Job
from scannerpy.stdlib import parsers

with Database() as db:
    videos = db.ingest_video_collection('my_videos', ['vid0.mp4', 'vid1.mkv'])
    frame = videos.as_op().all()
    histograms = db.ops.Histogram(frame = frame, device=DeviceType.GPU)
    job = Job(columns = [histograms], name = 'my_videos_hist')
    output = db.run(job)
    vid0_hists = output.tables(0).load(['histogram'], parsers.histograms)
```

[Click here to see more code examples of using Scanner.](https://github.com/scanner-research/scanner/tree/master/examples/tutorial)

Scanner makes it easy to use existing computer vision and pixel processing tools. For example, Scanner supports deep neural networks with [Caffe](https://github.com/scanner-research/scanner/tree/master/examples/caffe), image processing with [OpenCV](https://github.com/scanner-research/scanner/blob/master/examples/opticalflow/flow.py) and [Halide](https://github.com/scanner-research/scanner/tree/master/examples/halide), and object tracking with Struck.

Scanner is an active research project, part of a collaboration between Carnegie Mellon and Stanford. Please contact [Alex Poms](https://github.com/apoms) and [Will Crichton](https://github.com/willcrichton) with questions.

## Quick start ##

To quickly dive into Scanner, you can use one of our prebuilt [Docker images](https://hub.docker.com/r/scannerresearch/scanner). Start by [installing Docker](https://docs.docker.com/engine/installation/#supported-platforms).

If you have a GPU and you're running on Linux, install [nvidia-docker](https://github.com/NVIDIA/nvidia-docker) and run:

```bash
pip install --upgrade nvidia-docker-compose
wget http://raw.githubusercontent.com/scanner-research/scanner/master/docker-compose.yml
nvidia-docker-compose pull
nvidia-docker-compose run --service-ports gpu /bin/bash
```

Otherwise, you should run:


```bash
pip install --upgrade docker-compose
wget http://raw.githubusercontent.com/scanner-research/scanner/master/docker-compose.yml
docker-compose pull
docker-compose run --service-ports cpu /bin/bash
```

Then inside your Docker container, run:

```bash
cd examples
jupyter notebook --allow-root --ip=0.0.0.0 --port=8888
```

Then in your browser, go to [http://localhost:8888](http://localhost:8888) and copy in the token from the Jupyter console logs. Click "Walkthrough.ipynb" and follow the directions in the notebook.

## Learning Scanner ##

To get started building your own applications with Scanner, check out:

* [Build instructions](https://github.com/scanner-research/scanner/wiki/Building-Scanner)
* [Tutorials](https://github.com/scanner-research/scanner/wiki/Getting-started)
* [Documentation](https://github.com/scanner-research/scanner/wiki/Documentation)

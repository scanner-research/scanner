# Scanner

Scanner is a system for low-level, high-performance batch processing of images and videos, or visual data. It lets you write functions that get mapped across batches of frames. These functions can execute on a multi-core CPU or GPU, and can be distributed across multiple machines. Scanner also includes a library of components for simplifying the handling of visual data:

* Parallel video decode/encode on software or dedicated hardware
* [Caffe](https://github.com/bvlc/caffe) support for neural network evaluation
* [OpenCV](https://github.com/opencv/opencv) support with included kernels for color histograms and optical flow
* Object tracking in videos with [Struck](https://github.com/samhare/struck)

Additionally, Scanner offers several utilities for ease of development:

* Profiling via [chrome://tracing](https://www.chromium.org/developers/how-tos/trace-event-profiling-tool)
* Support for different storage backends including [Google Cloud Storage](https://cloud.google.com/storage/)
* Python interface for extracting results from the database
* Server and web interface for visualizing Scanner query results

Scanner is an active research project, part of a collaboration between Carnegie Mellon and Stanford. Please contact [Alex Poms](https://github.com/apoms) and [Will Crichton](https://github.com/willcrichton) with questions.

TODO: have code examples

## Building

You will need [OpenCV](https://github.com/opencv/opencv) (2.4.x or 3.x, 3.x preferred) built with [opencv_contrib](https://github.com/opencv/opencv_contrib/).

### OS X Dependencies
```
brew install openssl curl webp homebrew/science/opencv3 ffmpeg mpich
```
### Ubuntu Dependencies

```
sudo apt-get install -y libopenssl-dev libcurl3-dev liblzma-dev libprotobuf-dev protobuf-compiler \
     libeigen3-dev libgflags-dev libgoogle-glog-dev libatlas-base-dev libsuitesparse-dev \
     libx264-dev libavcodec-dev libavresample-dev libavformat-dev libavfilter-dev ffmpeg \
     libbpng-dev libjpeg-dev libbz2-dev cmake
```

### Python dependencies

```
pip install numpy protobuf toml
```

### Building the project

```
git clone https://github.com/apoms/scanner && cd scanner
mkdir build && cd build
cmake -D PIPELINE_FILE=../scanner/pipelines/sample_pipeline.cpp ..
```

## Running Scanner

Building Scanner produces one main executable `scanner_server` which manages data and executes queries. Scanner also comes with several Python scripts for managing configuration files and piping data in and out of Scanner.

### Setup

Run `python scripts/setup.py` to configure Scanner for your machine. This creates a configuration file `~/.scanner.toml` used by default in all Scanner jobs.

### Ingest
```
./build/scanner_server ingest <datasetName> <path/to/videoFile.txt>
```

Ingest takes a newline-separated textfile list of images and videos and copies them into the Scanner database, creating a new dataset from the collection. For example, if you run `ingest movies videos.txt` where `videos.txt` has the format:

```
/home/wcrichto/meanGirls.mp4
/home/wcrichto/theBourneIdentity.mp4
```

Then Scanner will create a dataset called `movies` which contains the two videos in the text file.

### Run
```
./build/scanner_server run <jobName> <datasetName>
```

Run will execute the pipeline you specified with `-D PIPELINE_FILE=...` to cmake over the given dataset `datasetName`. The results of that exeuction, or job, will be named `jobName`. For example, if you compile with `-D PIPELINE_FILE=../scanner/pipelines/opticalflow_viz_pipeline.cpp` and execute `run movieflow movies`, then it will compute an optical flow visualization for each frame of both movies we ingested earlier and save the results to disk.

To extract the results from the Scanner database, look at the Python documentation.

### Rm
```
./build/scanner_server rm <job|dataset> <name>
```

Rm will remove an object with the given `name` of the given type, either a `job` or a `dataset`, from the database. For example, if you run `rm job movieflow`, it will delete the results of the optical flow job we ran earlier.

### Python interface

The file `scripts/scanner.py` provides a `Scanner` class that lets you load raw byte buffers output from a Scanner job. For example, to load our `movieflow` job from earlier:

```
from scanner import Scanner
import numpy as np

db = Scanner()

@db.loader('opticalflow')
def load_opticalflow(buf, metadata)
    return np.frombuffer(buf, dtype=np.dtype(np.float32)) \
             .reshape((metadata.width, metadata.height, 2))
             
for video in load_opticalflow('movieflow', 'movies'):
    print("Processing {}".format(video['path']))
    do_something_with_flow_vectors(video['buffers'])
```

In this example, the `@db.loader` decorator takes a column name to load, e.g. `'opticalflow`', and a function that converts a single output and returns a Python-understandable data type. Here, the output of Scanner's optical flow component returns a `W x H x 2` matrix where `W x H` are the dimensions of the input video, and each pixel contains a 2-D vector corresponding to its velocity from the previous frame. We convert the raw bytes into a numpy matrix using `numpy.frombuffer`.

When you call the decorated function with a particular job name and dataset name, it fetches and decodes all the results in that column for that job on that dataset, shown in the for loop above.

## Building the results server
Enable the CMake flag `-DBUILD_SERVER=ON`.

### OS X Dependencies
#### Installing Folly
https://github.com/facebook/folly

Last I checked, the Homebrew formula does not work correctly with proxygen.
#### Installing Wangle
https://github.com/facebook/wangle
#### Installing Proxygen
https://dalzhim.wordpress.com/2016/04/27/compiling-facebooks-proxygen-on-os-x/

### Ubuntu Dependencies

#### Installing Proxygen
https://github.com/facebook/proxygen
```
git clone https://github.com/facebook/proxygen
cd proxygen/proxygen
sudo ./deps.sh && sudo ./reinstall
```

## Building UI

### Ubuntu
```
sudo apt-get install npm nodejs
sudo ln -s `which nodejs` /usr/bin/node
cd www
npm install
./node_modules/webpack/bin/webpack.js
```

### OS X
```
brew install npm
cd www
npm install
./node_modules/webpack/bin/webpack.js
```

## Contributing

Before committing, please install the Git hooks for this project.

```
git clone https://github.com/willcrichton/scanner-hooks
<edit files>
./scanner-hooks/install_hooks.sh </path/to/scanner>
```

You'll need to edit `pre-commit-clang-format` and change `CLANG_FORMAT` to point your `clang-format` binary. Must be version 4.0 or above.

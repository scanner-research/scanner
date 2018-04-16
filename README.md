# Scanner: Efficient Video Analysis at Scale [![Build Status](https://travis-ci.org/scanner-research/scanner.svg?branch=master)](https://travis-ci.org/scanner-research/scanner) #

Scanner is a system for efficient video processing and understanding at scale.
Scanner provides a python API for expressing computations and a heterogeneous
runtime for scheduling these computations onto clusters of machines with
CPUs or GPUs.

* [Install](https://github.com/scanner-research/scanner#install)
* [Running Scanner](https://github.com/scanner-research/scanner#running-scanner)
* [Tutorials & Examples](https://github.com/scanner-research/scanner#tutorials--examples)
* [Documentation](https://github.com/scanner-research/scanner#documentation)

Scanner is an active research project, part of a collaboration between Carnegie Mellon and Stanford. Please contact [Alex Poms](https://github.com/apoms) and [Will Crichton](https://github.com/willcrichton) with questions.

## Installation

There are two ways to build and run Scanner on your machine: using Docker, or building from source.

### Docker

We provide prebuilt [Docker](https://docs.docker.com/engine/installation/#supported-platforms) images containing Scanner and all its dependencies (e.g. OpenCV, Caffe) at [`scannerresearch/scanner`](https://hub.docker.com/r/scannerresearch/scanner/). We support the following builds:
* `scannerresearch/scanner:cpu` - CPU-only build
* `scannerresearch/scanner:gpu-8.0-cudnn6` - CUDA 9.1, CUDNN 7
* `scannerresearch/scanner:gpu-8.0-cudnn6` - CUDA 9.0, CUDNN 7
* `scannerresearch/scanner:gpu-8.0-cudnn6` - CUDA 8.0, CUDNN 6

### From Source

Follow the instructions at [INSTALL](https://github.com/scanner-research/scanner/blob/master/INSTALL.md)
to build Scanner from source. To start processing some videos, check out [Running Scanner](https://github.com/scanner-research/scanner#running-scanner).

## Getting started

To start using Scanner, we recommend trying our Jupyter notebook tutorial. To start the notebook, if you're using Docker:

```bash
pip install --upgrade docker-compose
wget https://raw.githubusercontent.com/scanner-research/scanner/master/docker/docker-compose.yml
docker-compose up cpu
```

If you installed Scanner yourself, then run:

```bash
cd path/to/scanner
jupyter notebook --ip=0.0.0.0 --port=8888
```

Then visit port 8888 on your server/localhost, click through to `examples/Walkthrough.ipynb`, and follow the directions in the notebook. To learn more, the tutorials and examples are located in the
[examples](https://github.com/scanner-research/scanner/tree/master/examples)
directory. You can find a comprehensive API reference in the [documentation](https://scanner-research.github.io/scanner/index.html)

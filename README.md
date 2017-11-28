# Scanner: Efficient Video Analysis at Scale [![Build Status](https://travis-ci.org/scanner-research/scanner.svg?branch=master)](https://travis-ci.org/scanner-research/scanner) #

Scanner is a system for efficient video processing and understanding at scale.
Scanner provides a python API for expressing computations and a heterogeneous
runtime for scheduling these computations onto clusters of machines with
CPUs or GPUs.

* [Install](https://github.com/scanner-research/scanner#install)
* [Running Scanner](https://github.com/scanner-research/scanner#running-scanner)
* [Tutorials & Examples](https://github.com/scanner-research/scanner#tutorials--examples)
* [Documentation](https://github.com/scanner-research/scanner#documentation)
* [Contributing](https://github.com/scanner-research/scanner#contributing)

Scanner is an active research project, part of a collaboration between Carnegie Mellon and Stanford. Please contact [Alex Poms](https://github.com/apoms) and [Will Crichton](https://github.com/willcrichton) with questions.

## Install

There are two ways to build and run Scanner on your machine:
* [Docker](https://github.com/scanner-research/scanner#docker)
* [From Source](https://github.com/scanner-research/scanner#from-source)

### Docker
First, install [Docker](https://docs.docker.com/engine/installation/#supported-platforms).
If you have a GPU and you're running on Linux, install [nvidia-docker](https://github.com/NVIDIA/nvidia-docker) and run:

```bash
pip install --upgrade nvidia-docker-compose
wget http://raw.githubusercontent.com/scanner-research/scanner/master/docker-compose.yml
nvidia-docker-compose pull gpu
nvidia-docker-compose run --service-ports gpu /bin/bash
```

Otherwise, you should run:

```bash
pip install --upgrade docker-compose
wget http://raw.githubusercontent.com/scanner-research/scanner/master/docker-compose.yml
docker-compose pull cpu
docker-compose run --service-ports cpu /bin/bash
```

If these commands were successful, you should now have bash session at the
Scanner directory inside the docker container. To start processing some videos,
check out [Running Scanner](https://github.com/scanner-research/scanner#running-scanner)

### From Source
Follow the instructions at [Build Instructions](https://github.com/scanner-research/scanner/wiki/Building-Scanner-from-Source)
to build Scanner from source. To start processing some videos, check out [Running Scanner](https://github.com/scanner-research/scanner#running-scanner).

## Running Scanner

Since Scanner programs are written using a high-level python API, running a
Scanner program is as simple as executing a python script. Let's run a Scanner
job now to find all the faces of people in a video (you can also use your own 
video if you have one on-hand). Run the following commands:

```bash
cd path/to/your/scanner/directory
# Download an example video (or use your own)
wget https://storage.googleapis.com/scanner-data/tutorial_assets/star_wars_heros.mp4
# Run the Scanner program
python examples/face_detection/main.py star_wars_heros.mp4

```

You should see several progress bars indicating the video is being processed.
When finished, there will be an mp4 file in your current directory called `
star_wars_heros_faces.mp4` with bounding boxes drawn over every
face in the original video. Congratulations, you just ran your first Scanner
program! Here's a few next steps:

* To learn how to start writing your own Scanner programs, dive into the API with the [tutorials](https://github.com/scanner-research/scanner#tutorials--examples).
* To run other Scanner programs on your videos, check out the [examples](https://github.com/scanner-research/scanner#tutorials--examples).
* If you're looking for a code reference, check out the [documentation](https://github.com/scanner-research/scanner#documentation)

## Tutorials & Examples

The tutorials and examples are located in the
[examples](https://github.com/scanner-research/scanner/tree/master/examples)
directory. Some of the examples include:

* [Locate and recognize faces in a video](https://github.com/scanner-research/scanner/blob/master/examples/face_detection/detect.py)
* [Detect shots in a film](https://github.com/scanner-research/scanner/blob/master/examples/shot_detection/shot_detect.py)
* [Search videos by image](https://github.com/scanner-research/scanner/blob/master/examples/reverse_image_search/search.py)

## Documentation & How-To's

TODO(apoms)

## Contributing

TODO(apoms)

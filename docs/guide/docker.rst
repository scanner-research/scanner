.. _docker:

Docker
======

`Docker <https://docs.docker.com>`__ is a service for managing containers, which you can think of as lightweight virtual machines. If you want to run Scanner in a distributed setting (e.g. on a cloud platform), Docker is essential for providing a consistent runtime environment on your worker machines, but it's also useful for testing locally to avoid having to install all of Scanner's dependencies. We provide prebuilt Docker images containing Scanner and all its dependencies (e.g. OpenCV, Caffe) at `scannerresearch/scanner <https://hub.docker.com/r/scannerresearch/scanner/>`__.

To start using Scanner with Docker, first install `Docker <https://docs.docker.com/install/>`__. If you have a GPU and you're running on Linux, you can install `nvidia-docker <https://github.com/NVIDIA/nvidia-docker>`__ (which provides GPU support inside Docker containers). Then run:

.. code-block:: bash

   pip3 install --upgrade docker-compose
   wget https://raw.githubusercontent.com/scanner-research/scanner/master/docker/docker-compose.yml
   docker-compose run --service-ports cpu /bin/bash

If you installed nvidia-docker, then use :code:`gpu` intead of :code:`cpu` in the above :code:`docker-compose` commands.

This installs the `docker-compose <https://docs.docker.com/compose/overview/>`__ utility which helps manage Docker containers. It uses the :code:`docker-compose.yml` configuration file to create an instance of the Scanner docker image.

If these commands were successful, you should now have bash session inside the docker container. To start using Scanner to process videos, check out :ref:`getting-started`.

The full set of docker configurations we provide are:

- :code:`scannerresearch/scanner:cpu-VERSION` - CPU-only build
- :code:`scannerresearch/scanner:gpu-9.1-cudnn7-VERSION` - CUDA 9.1, CUDNN 7
- :code:`scannerresearch/scanner:gpu-8.0-cudnn7-VERSION` - CUDA 8.0, CUDNN 7
- :code:`scannerresearch/scanner:gpu-8.0-cudnn6-VERSION` - CUDA 8.0, CUDNN 6

where :code:`VERSION` is one of:

- :code:`latest` - The most recent build of the master branch
- :code:`vX.X.X` - A git tag (where X is an integer)

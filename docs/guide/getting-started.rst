.. _getting-started:

Installation
------------

There are several options for installing Scanner:

- :ref:`docker`: The easiest way to get started with Scanner is using our pre-built :ref:`docker` images.

- :ref:`homebrew`: On MacOS, you can install using the unofficial package manager Homebrew.

- :ref:`source`: Scanner has out-of-the-box compatibility with frameworks like OpenCV and Caffe, but the flip side is that installing all of the dependencies can take a long time, which is why we recommend Docker for users just getting started.

.. _docker:

Docker
******

`Docker <https://docs.docker.com>`__ is a service for managing containers, which you can think of as lightweight virtual machines. If you want to run Scanner in a distributed setting (e.g. on a cloud platform), Docker is essential for providing a consistent runtime environment on your worker machines, but it's also useful for testing locally to avoid having to install all of Scanner's dependencies. We provide prebuilt Docker images containing Scanner and all its dependencies (e.g. OpenCV) at `scannerresearch/scanner <https://hub.docker.com/r/scannerresearch/scanner/>`__. We also provide Docker images containing Scanner along with all the standard library kernels in Scannertools at  `scannerresearch/scannertools <https://hub.docker.com/r/scannerresearch/scannertools/>`__.

To start using Scanner with Docker, first install `Docker <https://docs.docker.com/install/>`__. If you have a GPU and you're running on Linux, you can install `nvidia-docker <https://github.com/NVIDIA/nvidia-docker>`__ (which provides GPU support inside Docker containers). Then run:

.. code-block:: bash

   pip3 install --upgrade docker-compose
   wget https://raw.githubusercontent.com/scanner-research/scanner/master/docker/docker-compose.yml
   docker-compose run --service-ports cpu /bin/bash

If you installed nvidia-docker, then use :code:`gpu` intead of :code:`cpu` in the above :code:`docker-compose` commands.

This installs the `docker-compose <https://docs.docker.com/compose/overview/>`__ utility which helps manage Docker containers. It uses the :code:`docker-compose.yml` configuration file to create an instance of the Scanner docker image.

The full set of docker configurations we provide are:

- :code:`scannerresearch/scannertools:cpu-VERSION` - CPU-only build
- :code:`scannerresearch/scannertools:gpu-9.1-cudnn7-VERSION` - CUDA 9.1, CUDNN 7
- :code:`scannerresearch/scannertools:gpu-9.0-cudnn7-VERSION` - CUDA 9.0, CUDNN 7
- :code:`scannerresearch/scannertools:gpu-8.0-cudnn6-VERSION` - CUDA 8.0, CUDNN 6

where :code:`VERSION` is one of:

- :code:`latest` - The most recent build of the master branch
- :code:`vX.X.X` - A git tag (where X is an integer)



.. _homebrew:

Homebrew
********

Homebrew installation for Scanner is coming soon. Check back in a couple weeks!

.. _source:

From Source
***********
Building Scanner from source is a bit more involved than the other installation options. Check out the :ref:`from_source` page for more information.

.. toctree::
   :hidden:

   from_source

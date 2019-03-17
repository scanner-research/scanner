.. _quickstart:

Quickstart
==========

If you want to try Scanner out on one of your own videos as quickly as possible, install `Docker <https://docs.docker.com/install/>`__ (if you have a GPU and you're running on Linux, you can also install `nvidia-docker <https://github.com/NVIDIA/nvidia-docker>`__ which provides GPU support inside Docker containers). Then run:

.. code-block:: bash

   pip3 install --upgrade docker-compose
   wget https://raw.githubusercontent.com/scanner-research/scanner/master/docker/docker-compose.yml
   docker-compose run --service-ports cpu /bin/bash

Now you can run any of the example applications on your video:

.. code-block:: bash

   cd examples/apps/quickstart
   wget https://storage.googleapis.com/scanner-data/public/sample-clip.mp4
   python3 main.py

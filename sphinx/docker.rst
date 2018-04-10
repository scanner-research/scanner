Docker
------

First, install `Docker <https://docs.docker.com/install/>`_. If you have a GPU
and you're running on Linux, install
`nvidia-docker <https://github.com/NVIDIA/nvidia-docker>`_ and run:

.. code-block:: bash

   pip install --upgrade nvidia-docker-compose
   wget https://raw.githubusercontent.com/scanner-research/scanner/master/docker/docker-compose.yml
   nvidia-docker-compose pull gpu
   nvidia-docker-compose run --service-ports gpu /bin/bash

Otherwise, you should run:

.. code-block:: bash

   pip install --upgrade docker-compose
   wget https://raw.githubusercontent.com/scanner-research/scanner/master/docker/docker-compose.yml
   docker-compose pull cpu
   docker-compose run --service-ports cpu /bin/bash

If these commands were successful, you should now have bash session at the Scanner directory inside the docker container. To start processing some videos, check out Running Scanner

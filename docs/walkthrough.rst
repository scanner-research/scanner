.. _walkthrough:

Interactive Jupyter Walkthrough
===============================

To get a more detailed understanding of how Scanner can be used in a real
application, we recommend trying our Jupyter notebook tutorial. To start the
notebook, if you're using Docker:

.. code-block:: bash

   pip install --upgrade docker-compose
   wget https://raw.githubusercontent.com/scanner-research/scanner/master/docker/docker-compose.yml
   docker-compose up cpu

If you installed Scanner yourself, then run:

.. code-block:: bash

   cd path/to/scanner
   jupyter notebook --ip=0.0.0.0 --port=8888


Then visit port 8888 on your server/localhost, click through to
:code:`examples/Walkthrough.ipynb`, and follow the directions in the notebook.

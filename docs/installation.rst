.. _installation:

Installation
============

Scanner has out-of-the-box compatibility with frameworks like OpenCV and Caffe, but the flip side is that installing all of the dependencies can take a long time. The easiest way to get started with Scanner is using our pre-built :ref:`docker` images, but we also support :ref:`from_source`.

On MacOS, you can install Scanner using homebrew with the following commands:

.. code-block:: bash

   brew tap scanner-research/homebrew-scanner
   brew install scanner
   pip3 install scannerpy


.. toctree::
   :maxdepth: 1

   docker
   from_source

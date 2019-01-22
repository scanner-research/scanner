.. _quickstart:

Quickstart
==========

To explain how Scanner is used, let's walk through a simple example that reads every third frame from a video, resizes the frames, and then creates a new video from the sequence of resized frames.

.. note::

   This Quickstart walks you through a very basic Scanner application that downsamples a video in space and time. Once you are done with this guide, check out the `examples <https://github.com/scanner-research/scanner/blob/master/examples>`__ directory for more useful applications, such as using Tensorflow `for detecting objects in all frames of a video <https://github.com/scanner-research/scanner/blob/master/examples/apps/object_detection_tensorflow>`__ and Caffe for `face detection <https://github.com/scanner-research/scanner/blob/master/examples/apps/face_detection>`__.

To run the code discussed here, install Scanner (:ref:`installation`). Then from the top-level Scanner directory, run:

.. code-block:: bash

   cd examples/apps/quickstart
   wget https://storage.googleapis.com/scanner-data/public/sample-clip.mp4
   python3 main.py

After :code:`main.py` exits, you should now have a resized version of :code:`sample-clip.mp4` named :code:`sample-clip-resized.mp4` in the current directory. Let's see how that happened by looking inside :code:`main.py`.

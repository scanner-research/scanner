API Reference
=============

Scanner has three main components to its API:

* The :ref:`Scanner Python API <scannerpy-docs>`, for defining/executing computation graphs and Python kernels
* The `Scanner C++ API </cpp/>`_, for defining C++ kernels
* The :ref:`Scannertools API <scannertools-docs>`, a standard library of premade kernels

.. _scannerpy-docs:

scannerpy - the main scanner API
--------------------------------

* :any:`scannerpy.client`: entrypoint for running computation graphs, similar to TensorFlow Session
* :any:`scannerpy.kernel`: defining custom Python kernels
* :any:`scannerpy.storage`: defining custom inputs/outputs to Scanner graphs
* :any:`scannerpy.kube`: Kubernetes API
* :any:`scannerpy.profiler`: handle to profiling data output by Scanner

.. _scannertools-docs:

scannertools - the Scanner standard library
-------------------------------------------

.. toctree::
   :maxdepth: 3

   api/scannertools

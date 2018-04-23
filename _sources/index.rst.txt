.. scanner documentation master file, created by
   sphinx-quickstart on Sun Nov 26 19:06:21 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Scanner
=======

Scanner is a system for writing applications that process video efficiently.

- **Computation Graphs:** Scanner applications are written by composing together functions that process streams of data (called Ops) into graphs. The Scanner runtime is then responsible for executing this graph efficiently given all the processing resources on your machine.
- **Random Access to Video:** Since Scanner understands how video is compressed, it can provide fast *random* access to video frames.
- **First-class Support for GPUs:** Most image processing algorithms can benefit greatly from GPUs, so Scanner provides first class support for writing Ops that execute on GPUs.
- **Distributed Execution:** Scanner can scale out applications to a cluster of machines.


.. toctree::
   :maxdepth: 2
   :includehidden:

   installation
   getting-started
   programming-guide
   api
   about

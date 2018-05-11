.. scanner documentation master file, created by
   sphinx-quickstart on Sun Nov 26 19:06:21 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. role:: python(code)
   :language: python

.. role:: bash(code)
   :language: bash

Scanner: Efficient Video Analysis at Scale
==========================================

.. raw:: html

   <a href="https://travis-ci.org/scanner-research/scanner">
       <img
           alt="https://secure.travis-ci.org/scanner-research/scanner.svg?branch=master"
           src="https://secure.travis-ci.org/scanner-research/scanner.svg?branch=master"
       />
   </a>

Scanner is a system for developing applications that efficiently process large video datasets. Scanner applications can run on a multi-core laptop, a server packed with multiple GPUs, or a large number of machines in the cloud. Scanner has been used for:

- **Labeling and data mining large video collections:** Scanner is in use at Stanford University as the compute engine for visual data mining applications that detect people, commercials, human poses, etc. in datasets as big as 70,000 hours of TV news (12 billion frames, 20 TB) or 600 feature length movies (106 million frames). 

- **VR Video synthesis:** scaling the `Surround 360 VR video stitching software <https://github.com/scanner-research/Surround360>`__, which processes fourteen 2048x2048 input videos to produce 8k stereo video output.

To learn more about Scanner, see the documentation below or read the SIGGRAPH
2018 Technical Paper: `"Scanner: Efficient Video Analysis at Scale" <http://graphics.stanford.edu/papers/scanner/scanner_sig18.pdf>`__ by Poms, Crichton, Hanrahan, and Fatahalian.

Key Features
------------

Scanner's key features include:

- **Computation graphs designed for video processing:** Similar to the execution model used by many modern ML frameworks, creating a Scanner application involves composing together functions (called Scanner Ops) into a dataflow graph. Scanner graphs process sequences of video frames. Scanner graphs support features useful for video processing, such as the ability to sparsely sample frames from a video, access to temporal sliding windows of frames, and propagate state across computations on successive frames (e.g., tracking). The Scanner runtime schedules computation graphs efficiently onto one or many machines.

- **Random access to video frames:** Since Scanner's video data store has first-class knowledge of video formats, it can provide fast *random* access to compressed video frames.  This feature has proven useful in video data analytics applications that want to access a sparse set of frames from a video.

- **First-class support for GPU acceleration:** Most image processing algorithms can benefit greatly from GPU execution, so Scanner provides first-class support for writing Ops that utilize GPU execution. Scanner also leverages specialized GPU hardware for video decoding when available.

- **Distributed execution:** Scanner can scale out applications to hundreds of machines, and is designed to be fault tolerant, so your applciations can use cheaper preemptible machines on cloud computing platforms.

What Scanner is not:

- Scanner **is not** a new system for implementing new high-performance image and video processing kernels from scratch.  However, Scanner can be used to create scalable video processing applications by composing kernels that already exist as part of popular libraries such as OpenCV, Caffe, TensorFlow, etc. or have been implemented in popular languages like Cuda or Halide.  

Paper citation
--------------
Scanner will appear in the proceedings of SIGGRAPH 2018 as `"Scanner: Efficient Video Analysis at Scale <http://graphics.stanford.edu/papers/scanner/scanner_sig18.pdf>`__ by Poms, Crichton, Hanrahan, and Fatahalian. If you use Scanner in your research, we'd appreciate it if you cite the paper.


.. toctree::
   :maxdepth: 2
   :includehidden:

   installation
   getting-started
   programming-handbook
   api
   about


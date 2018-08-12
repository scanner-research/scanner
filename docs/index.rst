.. scanner documentation master file, created by
   sphinx-quickstart on Sun Nov 26 19:06:21 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. image:: scanner_logo.png

===========================

Scanner is a system for developing applications that efficiently process large video datasets. Scanner applications can run on a multi-core laptop, a server packed with multiple GPUs, or a large number of machines in the cloud. Scanner has been used for:

- **Labeling and data mining large video collections:** Scanner is in use at Stanford University as the compute engine for visual data mining applications that detect faces, commercials, human poses, etc. in datasets as big as 70,000 hours of TV news (12 billion frames, 20 TB) or 600 feature length movies (106 million frames).

- **VR Video synthesis:** Scanner is in use at Facebook to scale the `Surround 360 VR video stitching software <https://github.com/scanner-research/Surround360>`__, which processes fourteen 2048x2048 input videos to produce 8k stereo video output.

To learn more about Scanner, see the documentation below or read the SIGGRAPH
2018 Technical Paper: `"Scanner: Efficient Video Analysis at Scale" <http://graphics.stanford.edu/papers/scanner/scanner_sig18.pdf>`__ by Poms, Crichton, Hanrahan, and Fatahalian.

For easy access to off-the-shelf pipelines built using Scanner like face detection and optical flow, check out our `scannertools <https://github.com/scanner-research/scannertools>`__ library.

Key Features
------------

Scanner's key features include:

- **Video processing computations as dataflow graphs**.  Like many modern ML frameworks, Scanner structures video analysis tasks as dataflow graphs whose nodes produce and consume sequences of per-frame data. Scanner's embodiment of the dataflow model includes operators useful for video processing tasks such as sparse frame sampling (e.g., "frames known to contain a face"), sliding window frame access (e.g., stencils for temporal smoothing), and stateful processing across frames (e.g., tracking).

- **Videos as logical tables** To simplify the management of and access to large-numbers of videos, Scanner represents video collections and the pixel-level products of video frame analysis (e.g., flow fields, depth maps, activations) as tables in a data store. Scanner's data store features first-class support for video frame column types to facilitate key performance optimizations, such as storing video in compressed form and providing fast access to sparse lists of video frames.

- **First-class support for GPU acceleration:** Since many video processing algorithms benefit from GPU acceleration, Scanner provides first-class support for writing dataflow graph operations that utilize GPU execution. Scanner also leverages specialized GPU hardware for video decoding when available.

- **Fault tolerant, distributed execution:** Scanner applications can be run on the cores of a single machine, on a multi-GPU server, or scaled to hundreds of machines (potentially with heterogeneous numbers of GPUs), without significant source-level change.  Scanner also provides fault tolerance, so your applications can not only utilize many machines, but use cheaper preemptible machines on cloud computing platforms.


What Scanner **is not**:

Scanner is not a system for implementing new high-performance image and video processing kernels from scratch.  However, Scanner can be used to create scalable video processing applications by composing kernels that already exist as part of popular libraries such as OpenCV, Caffe, TensorFlow, etc. or have been implemented in popular performance-oriented languages like `CUDA <https://developer.nvidia.com/cuda-zone>`__ or `Halide <http://halide-lang.org/>`__. Yes, you can write your own dataflow graph operations in Python or C++ too!

.. toctree::
   :maxdepth: 2
   :includehidden:

   installation
   getting-started
   programming-handbook
   api
   about

# Scanner

_For build instructions, tutorials, documentation, and contributing guidelines, visit the [Scanner wiki](https://github.com/apoms/scanner/wiki)._

Scanner is a system for low-level, high-performance batch processing of images and videos, or visual data. It lets you write functions that get mapped across batches of frames. These functions can execute on a multi-core CPU or GPU, and can be distributed across multiple machines. Scanner also includes a library of components for simplifying the handling of visual data:

* Parallel video decode/encode on software or dedicated hardware
* [Caffe](https://github.com/bvlc/caffe) support for neural network evaluation
* [OpenCV](https://github.com/opencv/opencv) support with included kernels for color histograms and optical flow
* Object tracking in videos with [Struck](https://github.com/samhare/struck)

Additionally, Scanner offers several utilities for ease of development:

* Profiling via [chrome://tracing](https://www.chromium.org/developers/how-tos/trace-event-profiling-tool)
* Support for different storage backends including [Google Cloud Storage](https://cloud.google.com/storage/)
* Python interface for extracting results from the database
* Server and web interface for visualizing Scanner query results

Scanner is an active research project, part of a collaboration between Carnegie Mellon and Stanford. Please contact [Alex Poms](https://github.com/apoms) and [Will Crichton](https://github.com/willcrichton) with questions.


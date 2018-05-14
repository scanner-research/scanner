# Scanner: Efficient Video Analysis at Scale [![Build Status](https://travis-ci.org/scanner-research/scanner.svg?branch=master)](https://travis-ci.org/scanner-research/scanner) #

Scanner is a system for developing applications that efficiently process large video datasets. Scanner has been used for both video analysis and video snythesis tasks, such as:
* **Labeling and data mining large video collections:** Scanner is in use at Stanford University as the compute engine for visual data mining applications that detect people, commercials, human poses, etc. in datasets as big as 70,000 hours of TV news (12 billion frames, 20 TB) or 600 feature length movies (106 million frames).  We've used Scanner to run these tasks on hundreds of GPUs or thousands of CPUs on Google Compute Engine.
* **VR Video synthesis:** scaling the [Surround 360 VR video stitching software](https://github.com/scanner-research/Surround360) to 100's of CPUs. This application processes fourteen 2048x2048 input videos to produce 8k omidirectional stereo video output for VR display.

To learn more about Scanner, see the documentation below, check out our [various example applications](https://github.com/scanner-research/scanner/tree/master/examples), or read the SIGGRAPH 2018 Technical Paper: "[Scanner: Efficient Video Analysis at Scale](http://graphics.stanford.edu/papers/scanner/scanner_sig18.pdf)".

## Key Features

Scanner's key features include:

* **Video processing computations as dataflow graphs**.  Like many modern ML frameworks, Scanner structures video analysis tasks as dataflow graphs whose nodes produce and consume sequences of per-frame data. Scanner's embodiment of the dataflow model includes operators useful for video processing tasks such as sparse frame sampling (e.g., "frames known to contain a face"), sliding window frame access (e.g., stencils for temporal smoothing), and stateful processing across frames (e.g., tracking).

* **Videos as logical tables** To simplify the management of and access to large-numbers of videos, Scanner represents video collections and the pixel-level products of video frame analysis (e.g., flow fields, depth maps, activations) as tables in a data store. Scanner's data store features first-class support for video frame column types to facilitate key performance optimizations, such as storing video in compressed form and providing fast access to sparse lists of video frames. 

* **First-class support for GPU acceleration:** Since many video processing algorithms benefit from GPU acceleration, Scanner provides first-class support for writing dataflow graph operations that utilize GPU execution. Scanner also leverages specialized GPU hardware for video decoding when available.

* **Fault tolerant, distributed execution:** Scanner applications can be run on the cores of a single machine, on a multi-GPU server, or scaled to hundreds of machines (potentially with heterogeneous numbers of GPUs), without significant source-level change.  Scanner also provides fault tolerance, so your applications can not only utilize many machines, but use cheaper preemptible machines on cloud computing platforms.

What Scanner __is not__:

Scanner is not a system for implementing new high-performance image and video processing kernels from scratch.  However, Scanner can be used to create scalable video processing applications by composing kernels that already exist as part of popular libraries such as OpenCV, Caffe, TensorFlow, etc. or have been implemented in popular performance-oriented languages like [Cuda](https://developer.nvidia.com/cuda-zone) or [Halide](http://halide-lang.org/).  Yes, you can write your dataflow graph operations in Python or C++ too!

## Documentation

Scanner's documentation is hosted at [scanner.run](http://scanner.run). Here
are a few links to get you started:

* [Installation](http://scanner.run/installation.html)
* [Getting Started](http://scanner.run/getting-started.html)
* [Programming Handbook](http://scanner.run/programming-handbook.html)
* [API Reference](http://scanner.run/api.html)
* [SIGGRAPH 2018 Technical Paper](http://graphics.stanford.edu/papers/scanner/scanner_sig18.pdf)
* [Scanner Examples](https://github.com/scanner-research/scanner/tree/master/examples)

## Example code

Scanner applications are written using our Python API. Here's an example
application that resizes every third frame from a video and then saves the result as an mp4 video (our
[Quickstart](http://scanner.run/quickstart.html) walks through this
example in more detail):

```python
from scannerpy import Database, Job

# Ingest a video into the database (create a table with a row per video frame)
db = Database()
db.ingest_videos([('example_table', 'example.mp4')])

# Define a Computation Graph
frame = db.sources.FrameColumn()                                    # Read input frames from database
sampled_frame = db.streams.Stride(input=frame, stride=3)            # Select every third frame
resized = db.ops.Resize(frame=sampled_frame, width=640, height=480) # Resize input frames
output_frame = db.sinks.Column(columns={'frame': resized})          # Save resized frames as new video

# Set parameters of computation graph ops
job = Job(op_args={
    frame: db.table('example_table').column('frame'), # Column to read input frames from
    output_frame: 'resized_example'                   # Table name for computation output
})

# Execute the computation graph and return a handle to the newly produced tables
output_tables = db.run(output=output_frame, jobs=[job], force=True)

# Save the resized video as an mp4 file
output_tables[0].column('frame').save_mp4('resized_video')
```

If you'd like to see other example applications written with Scanner, check
out our [Examples](https://github.com/scanner-research/scanner/tree/master/examples)
directory in this repository.

## Contributing

If you'd like to contribute to the development of Scanner, you should first
build Scanner [from source](http://scanner.run/from_source.html).

Please submit a pull-request rebased against the most recent version of the
master branch and we will review your changes to be merged. Thanks for
contributing!

### Running tests
You can run our full suite of tests by executing `make test` in the directory
you used to build Scanner. This will run both our C++ tests and our end-to-end
tests that verify the python API.

## About
Scanner is an active research project, part of a collaboration between Stanford and Carnegie Mellon University. Please contact [Alex Poms](https://github.com/apoms) and [Will Crichton](https://github.com/willcrichton) with questions.

Scanner was developed with the support of the NSF (IIS-1539069), the Intel Corporation (through the Intel Science and Technology Center for Visual Cloud Computing and the NSF/Intel VEC program), and by Google.

### Paper citation
Scanner will appear in the proceedings of SIGGRAPH 2018 as "[Scanner: Efficient Video Analysis at Scale](http://graphics.stanford.edu/papers/scanner/scanner_sig18.pdf)" by Poms, Crichton, Hanrahan, and Fatahalian. If you use Scanner in your research, we'd appreciate it if you cite the paper.

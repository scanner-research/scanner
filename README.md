# Scanner: Efficient Video Analysis at Scale [![Build Status](https://travis-ci.org/scanner-research/scanner.svg?branch=master)](https://travis-ci.org/scanner-research/scanner) #

Scanner is a system for writing applications that process video efficiently. Scanner has been used for:
* **Labeling and data mining large video collections:** Scanner is in use at Stanford University as the compute engine for visual data mining applications that detect people, commercials, human poses, etc. in datasets as big as 70,000 hours of TV news (12 billion frames, 20 TB) or 600 feature length movies (106 million frames). 
* **VR Video synthesis:** scaling the [Surround 360 VR video stitching software](https://github.com/scanner-research/Surround360), which processes fourteen 2048x2048 input videos to produce 8k stereo video output.

To learn more about Scanner and how it can been used, see the documentation below or read the SIGGRAPH 2018 Technical Paper: "[Scanner: Efficient Video Analysis at Scale](http://graphics.stanford.edu/papers/scanner/scanner_sig18.pdf)" by Poms, Crichton, Hanrahan, and Fatahalian.

## Key Features

Scanner's key features include:
* **Computation graphs designed for video processing:** Similar to the execution model used by many modern ML frameworks, Scanner applications are written by composing together functions (called Scanner Ops) into dataflow graphs. Computation graphs process sequences of video frames. Scanner graphs support features useful for video processing, such as the ability to sparsely sample video frames, access to temporal sliding windows of frames, and propagate state across computations on successive frames (e.g., tracking). The Scanner runtime schedules computation graphs efficiently on the processing resources of your machine, whether it be a multi-core laptop, a machine with multiple GPUs, or a large collection of machines in the cloud.
* **Random access to video frames:** Since Scanner understands how video is compressed, it can provide fast *random* access to video frames.  This feature has proven useful in video data analytics applications that want to access a sparse set of frames from a video.
* **First-class support for GPU acceleration:** Most image processing algorithms can benefit greatly from GPU execution, so Scanner provides first-class support for writing Ops that execute on multiple GPUs in a single machine. Scanner also leverage specialized GPU hardware for video decoding.
* **Distributed execution:** Scanner can scale out applications to hundreds of machines, and is designed to use cheaper preemptible machines on cloud computing platforms.

## Documentation

Scanner's documentation is hosted at [scanner.run](http://scanner.run). Here
are a few links to get you started:

* [Installation](http://scanner.run/installation.html)
* [Getting Started](http://scanner.run/getting-started.html)
* [Programming Handbook](http://scanner.run/programming-handbook.html)
* [API Reference](http://scanner.run/api.html)
* [SIGGRAPH 2018 Technical Paper](http://graphics.stanford.edu/papers/scanner/scanner_sig18.pdf)  

## Example code

Scanner applications are written using our python API. Here's an example
application that resizes a video and then saves it as an mp4 (our
[Quickstart](http://scanner.run/quickstart.html) walks through this
example in more detail):

```python
from scannerpy import Database, Job

# Ingest a video into the database
db = Database()
db.ingest_videos([('example_table', 'example.mp4')])

# Define a Computation Graph
frame = db.sources.FrameColumn() # Read from the database
resized = db.ops.Resize(frame=frame, width=640, height=480) # Resize input frame
output_frame = db.sinks.Column(columns={'frame': resized}) # Save resized frame

job = Job(op_args={
    frame: db.table('example_table').column('frame'), # Column to read input frames from
    output_frame: 'resized_example' # Name of output table
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

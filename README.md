# Scanner: Efficient Video Analysis at Scale [![Build Status](https://travis-ci.org/scanner-research/scanner.svg?branch=master)](https://travis-ci.org/scanner-research/scanner) #

Scanner is a system for writing applications that process video efficiently.

Scanner has been used for:
* **Big video data analysis:** labeling and data mining two large video datasets: a dataset containing over 600 feature length movies (106 million frames) and a dataset of 70,000 hours of TV news (12 billion frames, 20 TB).
* **VR Video synthesis:** scaling the [Surround 360 VR video stitching software](https://github.com/scanner-research/Surround360), which processes fourteen 2048x2048 input videos to produce 8k stereo video output.

## Key Features

Scanner's key features include:
* **Computation graphs:** Scanner applications are written by composing together functions that process streams of data (called Ops) into graphs. The Scanner runtime is then responsible for executing this graph efficiently given all the processing resources on your machine.
* **Random access to video:** Since Scanner understands how video is compressed, it can provide fast *random* access to video frames.
* **First-class support for GPUs:** Most image processing algorithms can benefit greatly from GPUs, so Scanner provides first class support for writing Ops that execute on GPUs.
* **Distributed execution:** Scanner can scale out applications to hundreds of machines.

## Documentation

Scanner's documentation is hosted at [scanner.run](http://scanner.run). Here
are a few links to get you started:

* [Installation](http://scanner.run/installation.html)
* [Getting Started](http://scanner.run/getting-started.html)
* [Programming Handbook](http://scanner.run/programming-handbook.html)
* [API Reference](http://scanner.run/api.html)

## Example code

Scanner applications are written using our python API. Here's an example
application that resizes a video and then saves it as an mp4 (our
[Quickstart](http://crissy.pdl.cmu.edu:4567/quickstart.html) walks through this
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
output_tables[0].column('frame').save_mp4('resized_video.mp4')
```

If you'd like to see other example applications written with Scanner, check
out our [Examples](https://github.com/scanner-research/scanner/tree/master/examples)
directory in this repository.

## Contributing

If you'd like to contribute to the development of Scanner, you should first
build Scanner [from source](http://crissy.pdl.cmu.edu:4567/from_source.html).

Please submit a pull-request rebased against the most recent version of the
master branch and we will review your changes to be merged. Thanks for
contributing!

### Running tests
You can run our full suite of tests by executing `make test` in the directory
you used to build Scanner. This will run both our C++ tests and our end-to-end
tests that verify the python API.

## About
Scanner is an active research project, part of a collaboration between Carnegie
Mellon and Stanford. Please contact [Alex Poms](https://github.com/apoms) and
[Will Crichton](https://github.com/willcrichton) with questions.

### Paper citation
Scanner will appear in the proceedings of SIGGRAPH 2018. If you use this
software in your research, please be sure to cite the paper.

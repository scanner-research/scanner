# Scanner

This assumes you have already built Scanner. See [BUILD.md](https://github.com/apoms/scanner/blob/master/BUILD.md) for instructions on doing so.

## Setup

Run `python scripts/setup.py` to configure Scanner for your machine. This creates a configuration file `~/.scanner.toml` used by default in all Scanner jobs.

## Scanner server

Building Scanner produces one main executable `scanner_server` which manages data and executes queries. Scanner also comes with several Python scripts for managing configuration files and piping data in and out of Scanner.

### Ingest
```bash
./build/scanner_server ingest <datasetName> <path/to/videoFile.txt>
```

Ingest takes a newline-separated textfile list of images and videos and copies them into the Scanner database, creating a new dataset from the collection. For example, if you run `ingest movies videos.txt` where `videos.txt` has the format:

```
/home/wcrichto/meanGirls.mp4
/home/wcrichto/theBourneIdentity.mp4
```

Then Scanner will create a dataset called `movies` which contains the two videos in the text file.

### Run
```bash
./build/scanner_server run <jobName> <datasetName>
```

Run will execute the pipeline you specified with `-D PIPELINE_FILE=...` to cmake over the given dataset `datasetName`. The results of that exeuction, or job, will be named `jobName`. For example, if you compile with `-D PIPELINE_FILE=../scanner/pipelines/opticalflow_pipeline.cpp` and execute `run movieflow movies`, then it will compute [optical flow](https://en.wikipedia.org/wiki/Optical_flow) for each frame of both movies we ingested earlier and save the results to disk.

To extract the results from the Scanner database, look at the Python documentation.

### Rm
```bash
./build/scanner_server rm <job|dataset> <name>
```

Rm will remove an object with the given `name` of the given type, either a `job` or a `dataset`, from the database. For example, if you run `rm job movieflow`, it will delete the results of the optical flow job we ran earlier.

## Python interface

The file `scripts/scanner.py` provides a `Scanner` class that lets you load raw byte buffers output from a Scanner job. For example, to load our `movieflow` job from earlier:

```python
from scanner import Scanner
import numpy as np

db = Scanner()

@db.loader('opticalflow')
def load_opticalflow(buf, metadata)
    return np.frombuffer(buf, dtype=np.dtype(np.float32)) \
             .reshape((metadata.width, metadata.height, 2))

for video in load_opticalflow('movieflow', 'movies'):
    print("Processing {}".format(video['path']))
    do_something_with_flow_vectors(video['buffers'])
```

In this example, the `@db.loader` decorator takes a column name to load, e.g. `'opticalflow`', and a function that converts a single output and returns a Python-understandable data type. Here, the output of Scanner's optical flow component returns a `W x H x 2` matrix where `W x H` are the dimensions of the input video, and each pixel contains a 2-D vector corresponding to its velocity from the previous frame. We convert the raw bytes into a numpy matrix using `numpy.frombuffer`.

When you call the decorated function with a particular job name and dataset name, it fetches and decodes all the results in that column for that job on that dataset, shown in the for loop above.

## Writing a pipeline

Each Scanner job executes a pipeline, which is a sequence of _evaluators_. An evaluator is a stateful function that takes a batch of inputs and produces an output for every input. For example, Scanner comes with an `OpticalFlowEvaluator` that takes a sequence of images as input and outputs the flow betwen them. Scanner also has a `CaffeEvaluator` that evaluates a neural network on an image and an `EncoderEvalautor` that takes a batch of frames and encodes them into a single video file. In a pipeline, the output of one evaluator becomes the input to the next. For example, to blur faces in a video, your pipeline might be `{DecoderEvaluator, FaceFinderEvaluator, BoxBlurEvalautor, EncoderEvaluator}`.

You can find some sample pipelines in `scanner/pipelines`. TODO: explain more

## Writing an evaluator

TODO: write this section

## Profiling a job

TODO: write this section

# Scanner examples

This directory contains simple examples and full applications that 
demonstrate how to use Scanner. We recommend starting with the
[tutorials](https://github.com/scanner-research/scanner/blob/master/examples/tutorial).

## Tutorials
* [Walkthrough.ipynb](https://github.com/scanner-research/scanner/blob/master/examples/Walkthrough.ipynb): an IPython notebook that goes through a simple application (shot detection) using Scanner.
* [tutorial](https://github.com/scanner-research/scanner/blob/master/examples/tutorial): a set of well-commented files exploring different Scanner features in code.

If you want to run the notebook yourself so that you can interactively edit the
code, run:

```bash
cd path/to/your/scanner/directory/
cd examples
jupyter notebook --allow-root --ip=0.0.0.0 --port=8888
```

Then in your browser, go to [http://localhost:8888/notebooks/Walkthrough.ipynb](http://localhost:8888/notebooks/Walkthrough.ipynb) and copy in the token from the console logs. Follow the instructions in the Jupyter notebook.

## Applications

* [face_detection](https://github.com/scanner-research/scanner/blob/master/examples/apps/face_detection): Detecting and recognizing faces in a video.
* [shot_detection](https://github.com/scanner-research/scanner/blob/master/examples/apps/shot_detection): Segmenting a video into shots. Same application as the walkthrough.
* [reverse_image_search](https://github.com/scanner-research/scanner/blob/master/examples/apps/reverse_image_search): Searching through a video by image.
* [depth_from_stereo](https://github.com/scanner-research/scanner/blob/master/examples/apps/depth_from_stereo): Computing a per-pixel depth image from two views of the same location.
* [hyperlapse](https://github.com/scanner-research/scanner/blob/master/examples/apps/hyperlapse): Creating stable timelapse videos with the [Hyperlapse](https://www.microsoft.com/en-us/research/publication/real-time-hyperlapse-creation-via-optimal-frame-selection/) algorithm.
* [optical_flow](https://github.com/scanner-research/scanner/blob/master/examples/apps/optical_flow): Using OpenCV to compute flow fields within a video.

## How-Tos
* [tensorflow](https://github.com/scanner-research/scanner/blob/master/examples/how-tos/tensorflow): How to expose [TensorFlow](https://www.tensorflow.org/) computations as Scanner graph operations
* [caffe](https://github.com/scanner-research/scanner/blob/master/examples/how-tos/caffe): How to use Caffe nets as Scanner graph operations
* [python_kernel](https://github.com/scanner-research/scanner/blob/master/examples/how-tos/python_kernel): How to implement Scanner graph ops in python 
* [halide](https://github.com/scanner-research/scanner/blob/master/examples/how-tos/halide): How to use [Halide](http://halide-lang.org/) kernels as Scanner graph Ops.

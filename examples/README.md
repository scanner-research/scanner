# Scanner Examples

This directory contains simple examples and full applications that 
demonstrate how to use Scanner. 

## Tutorials

* [Walkthrough.ipynb](https://github.com/scanner-research/scanner/blob/master/examples/Walkthrough.ipynb): an IPython notebook that goes through a simple application (shot detection) using Scanner.
* [List of Tutorials](https://github.com/scanner-research/scanner/blob/master/examples/tutorials): a set of well-commented files exploring different Scanner features in code.

If you want to run the notebook yourself so that you can interactively edit the
code, run:

```bash
cd path/to/your/scanner/directory/
cd examples
jupyter notebook --allow-root --ip=0.0.0.0 --port=8888
```

Then in your browser, go to [http://localhost:8888/notebooks/Walkthrough.ipynb](http://localhost:8888/notebooks/Walkthrough.ipynb) and copy in the token from the console logs. Follow the instructions in the Jupyter notebook.

## Example Applications

* [face_detection](https://github.com/scanner-research/scanner/blob/master/examples/apps/face_detection): Detect all faces in a video, and then render a new video overlaying the face bounding boxes on the video.  
* [shot_detection](https://github.com/scanner-research/scanner/blob/master/examples/apps/shot_detection): Segment a video into shots and then create a single image montage featuring one thumbnail for each shot. (Same application as the walkthrough.)
* [hyperlapse](https://github.com/scanner-research/scanner/blob/master/examples/apps/hyperlapse): Create a stable timelapse video using the [Hyperlapse](https://www.microsoft.com/en-us/research/publication/real-time-hyperlapse-creation-via-optimal-frame-selection/) algorithm.
* [optical_flow](https://github.com/scanner-research/scanner/blob/master/examples/apps/optical_flow): Use OpenCV to compute flow fields within a video.
* [object_detection_tensorflow](https://github.com/scanner-research/scanner/blob/master/examples/apps/object_detection_tensorflow): Use Tensorflow's SSD Mobilenet DNN to detect objects.
* [detectron](https://github.com/scanner-research/scanner/blob/master/examples/apps/detectron): Use the Detectron object detection API for Caffe2 to detect objects.
* [reverse_image_search](https://github.com/scanner-research/scanner/blob/master/examples/apps/reverse_image_search): Search through a video to look for a query frame.
* [depth_from_stereo](https://github.com/scanner-research/scanner/blob/master/examples/apps/depth_from_stereo): Compute a per-pixel depth image from two views of the same location.

## How-Tos

* [tensorflow](https://github.com/scanner-research/scanner/blob/master/examples/how-tos/tensorflow): How to expose [TensorFlow](https://www.tensorflow.org/) computations as Scanner graph operations.
* [caffe](https://github.com/scanner-research/scanner/blob/master/examples/how-tos/caffe): How to use Caffe nets as Scanner graph operations.
* [python_kernel](https://github.com/scanner-research/scanner/blob/master/examples/how-tos/python_kernel): How to implement Scanner graph ops in Python. 
* [halide](https://github.com/scanner-research/scanner/blob/master/examples/how-tos/halide): How to use [Halide](http://halide-lang.org/) kernels as Scanner graph operations.

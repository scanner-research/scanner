Detectron on Scanner
====================

Pre-requisites:
---------------

1. Install Caffe2. If you install from source, make sure to build with:

```bash
cmake .. \
  -DCMAKE_DISABLE_FIND_PACKAGE_Eigen3=TRUE \
  -DBUILD_CUSTOM_PROTOBUF=OFF \
  -DBUILD_TEST=OFF \
  -DPYTHON_INCLUDE_DIR=/usr/include/python3.5/ \
  -DPYTHON_EXECUTABLE=/usr/bin/python3 \
  -DPYTHON_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython3.5m.so
```

2. Install Detectron.

3. Build Scanner without Caffe Ops (Caffe and Caffe2 can not be in the same process):

```
cmake .. -DBUILD_CAFFE_OPS=OFF
```

Example Usage:
--------------

```bash
DETECTRON_PATH=...

CONFIG_PATH=$DETECTRON_PATH/configs/12_2017_baselines/e2e_mask_rcnn_R-101-FPN_2x.yaml
WEIGHTS_PATH='https://s3-us-west-2.amazonaws.com/detectron/35861858/12_2017_baselines/e2e_mask_rcnn_R-101-FPN_2x.yaml.02_32_51.SgT4y1cO/output/train/coco_2014_train:coco_2014_valminusminival/generalized_rcnn/model_final.pkl'

python3 main.py --weights-path $WEIGHTS_PATH --config-path $CONFIG_PATH --video-path example.mp4
```

This will output a video named `example_detected.mp4' overlaid with the network
detections.

.. note:

   Caffe2 currently crashes when the program is cleaning up. You might see an
   error related to CUDA at the end of execution. This is expected.

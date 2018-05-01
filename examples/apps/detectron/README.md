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

3. Build Scanner without Caffe Ops

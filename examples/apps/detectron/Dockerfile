FROM scannerresearch/scanner:gpu-9.1-cudnn7
WORKDIR /opt/detectron

RUN cd /opt/detectron && \
    git clone --recursive https://github.com/pytorch/pytorch.git && \
    cd pytorch && \
    git submodule update --init && \
    mkdir build && cd build && \
    cmake .. -DCMAKE_DISABLE_FIND_PACKAGE_Eigen3=TRUE \
             -DBUILD_CUSTOM_PROTOBUF=OFF \
             -DBUILD_TEST=OFF \
             -DPYTHON_INCLUDE_DIR=/usr/include/python3.5/ \
             -DPYTHON_EXECUTABLE=/usr/bin/python3 \
             -DPYTHON_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython3.5m.so \
             -DBUILD_CUSTOM_PROTOBUF=OFF && \
    make install -j
ENV PYTHONPATH /usr/local/lib/python3/dist-packages:${PYTHONPATH}

RUN cd /opt/detectron && \
    git clone https://github.com/cocodataset/cocoapi.git && \
    cd cocoapi/PythonAPI && \
    python3 setup.py build_ext install

RUN pip3 install -y pyyaml urllib2 matplotlib

RUN cd /opt/detectron && \
    git clone https://github.com/facebookresearch/detectron && \
    git fetch origin pull/110/head:py3 && \
    git checkout py3 && \
    ./python2_to_python3_conversion_automated.sh && \
    sed -i '83s/ref_md5sum/ref_md5sum.decode("utf-8")/' ../lib/utils/io.py && \
    cd detectron/lib && \
    python3 setup.py develop --user

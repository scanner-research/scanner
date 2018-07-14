FROM scannerresearch/scanner:gpu-9.1-cudnn7
WORKDIR /opt/openreid

RUN cd /opt/openreid && \
    git clone https://github.com/Cysu/open-reid.git && \
    cd open-reid && \
    pip3 install http://download.pytorch.org/whl/cu91/torch-0.4.0-cp35-cp35m-linux_x86_64.whl && \
    python3 setup.py install 

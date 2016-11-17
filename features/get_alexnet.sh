#!/bin/bash

CWD=`pwd`
DIR=`basename $CWD`

prog() {
    mkdir -p features/alexnet
    cd features/alexnet

    wget https://raw.githubusercontent.com/BVLC/caffe/master/models/bvlc_reference_caffenet/deploy.prototxt
    wget http://dl.caffe.berkeleyvision.org/bvlc_reference_caffenet.caffemodel

    wget -c http://dl.caffe.berkeleyvision.org/caffe_ilsvrc12.tar.gz
    tar -zxf caffe_ilsvrc12.tar.gz

    cd $CWD
}

if [[ "$DIR" != "scanner" ]] && [[ "$1" != "-f" ]];
then
    echo "Warning: you must run this script from the Scanner repo root, and I don't think you are."
    echo "Run this again with -f if you're sure."
else
    prog
fi

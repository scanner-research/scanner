#!/bin/bash

CWD=`pwd`
DIR=`basename $CWD`

prog() {
    mkdir -p features/googlenet
    cd features/googlenet

    wget https://raw.githubusercontent.com/BVLC/caffe/master/models/bvlc_googlenet/deploy.prototxt
    wget http://dl.caffe.berkeleyvision.org/bvlc_googlenet.caffemodel
    cd $CWD
}

if [[ "$DIR" != "scanner" ]] && [[ "$1" != "-f" ]];
then
    echo "Warning: you must run this script from the Scanner repo root, and I don't think you are."
    echo "Run this again with -f if you're sure."
else
    prog
fi

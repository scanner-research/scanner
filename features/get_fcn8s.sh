#!/bin/bash

CWD=`pwd`
DIR=`basename $CWD`

prog() {
    mkdir -p features/fcn8s
    cd features/fcn8s

    wget http://dl.caffe.berkeleyvision.org/fcn8s-heavy-pascal.caffemodel
    wget https://raw.githubusercontent.com/shelhamer/fcn.berkeleyvision.org/master/voc-fcn8s/deploy.prototxt

    cd $CWD
}

if [[ "$DIR" != "scanner" ]] && [[ "$1" != "-f" ]];
then
    echo "Warning: you must run this script from the Scanner repo root, and I don't think you are."
    echo "Run this again with -f if you're sure."
else
    prog
fi

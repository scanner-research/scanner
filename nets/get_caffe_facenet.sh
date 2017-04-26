#!/bin/bash

CWD=`pwd`
DIR=`basename $CWD`

prog() {
    mkdir nets/caffe_facenet
    cd nets/caffe_facenet

    # Caffemodel 
    wget https://storage.googleapis.com/scanner-data/nets/caffe_facenet/facenet_deploy.caffemodel
    # Templates 
    wget https://storage.googleapis.com/scanner-data/nets/caffe_facenet/facenet_templates.bin
    # Prototxt 
    wget https://storage.googleapis.com/scanner-data/nets/caffe_facenet/facenet_deploy.prototxt

    cd $CWD
}

if [[ "$DIR" != "scanner" ]] && [[ "$1" != "-f" ]];
then
    echo "Warning: you must run this script from the Scanner repo root, and I don't think you are."
    echo "Run this again with -f if you're sure."
else
    prog
fi

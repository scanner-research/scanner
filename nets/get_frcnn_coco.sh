#!/bin/bash

CWD=`pwd`
DIR=`basename $CWD`

prog() {
    mkdir nets/faster_rcnn_coco
    cd nets/faster_rcnn_coco

    # Prototxt for MPI version
    wget https://storage.googleapis.com/scanner-data/models/faster_rcnn_coco/test.prototxt
    # Caffemodel for MPI
    wget https://storage.googleapis.com/scanner-data/models/faster_rcnn_coco/coco_vgg16_faster_rcnn_final.caffemodel

    cd $CWD
}

if [[ "$DIR" != "scanner" ]] && [[ "$1" != "-f" ]];
then
    echo "Warning: you must run this script from the Scanner repo root, and I don't think you are."
    echo "Run this again with -f if you're sure."
else
    prog
fi

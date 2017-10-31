#!/bin/bash

CWD=`pwd`
DIR=`basename $CWD`

prog() {
    mkdir -p nets/resnet
    cd nets/resnet

    wget https://storage.googleapis.com/scanner-data/models/resnet/ResNet-50-deploy.prototxt

    wget https://storage.googleapis.com/scanner-data/models/resnet/ResNet-50-model.caffemodel

    cd $CWD
}

if [[ "$DIR" != "scanner" ]] && [[ "$1" != "-f" ]];
then
    echo "Warning: you must run this script from the Scanner repo root, and I don't think you are."
    echo "Run this again with -f if you're sure."
else
    prog
fi

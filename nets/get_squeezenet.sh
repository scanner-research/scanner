#!/bin/bash

CWD=`pwd`
DIR=`basename $CWD`

prog() {
    mkdir features/squeezenet
    cd features/squeezenet

    wget https://raw.githubusercontent.com/DeepScale/SqueezeNet/master/SqueezeNet_v1.0/deploy.prototxt
    wget -O squeezenet_v1.0.caffemodel \
         https://github.com/DeepScale/SqueezeNet/blob/master/SqueezeNet_v1.0/squeezenet_v1.0.caffemodel?raw=true

    cd $CWD
}

if [[ "$DIR" != "scanner" ]] && [[ "$1" != "-f" ]];
then
    echo "Warning: you must run this script from the Scanner repo root, and I don't think you are."
    echo "Run this again with -f if you're sure."
else
    prog
fi

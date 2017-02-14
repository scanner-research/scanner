#!/bin/bash

CWD=`pwd`
DIR=`basename $CWD`

prog() {
    mkdir nets/cpm
    cd nets/cpm

    # Person center detection
    wget https://raw.githubusercontent.com/shihenw/convolutional-pose-machines-release/master/model/_trained_person_MPI/pose_deploy_copy_4sg_resize.prototxt
    wget http://pearl.vasc.ri.cmu.edu/caffe_model_github/model/_trained_person_MPI/pose_iter_70000.caffemodel

    # Pose estimation
    wget https://raw.githubusercontent.com/shihenw/convolutional-pose-machines-release/master/model/_trained_MPI/pose_deploy_resize.prototxt
    wget http://pearl.vasc.ri.cmu.edu/caffe_model_github/model/_trained_MPI/pose_iter_320000.caffemodel

    cd $CWD
}

if [[ "$DIR" != "scanner" ]] && [[ "$1" != "-f" ]];
then
    echo "Warning: you must run this script from the Scanner repo root, and I don't think you are."
    echo "Run this again with -f if you're sure."
else
    prog
fi

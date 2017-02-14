#!/bin/bash

CWD=`pwd`
DIR=`basename $CWD`

prog() {
    mkdir nets/cpm2
    cd nets/cpm2

    # Prototxt for COCO version
    wget https://raw.githubusercontent.com/CMU-Perceptual-Computing-Lab/caffe_demo/master/model/coco/pose_deploy_linevec.prototxt
    mv pose_deploy_linvec.prototxt coco_pose_deploy_linvec.prototxt
    # Prototxt for MPI version
    wget https://raw.githubusercontent.com/CMU-Perceptual-Computing-Lab/caffe_demo/master/model/mpi/pose_deploy_linevec.prototxt
    mv pose_deploy_linvec.prototxt mpi_pose_deploy_linvec.prototxt
    # Caffemodel for COCO
    wget http://posefs1.perception.cs.cmu.edu/Users/tsimon/Projects/coco/data/models/coco/pose_iter_440000.caffemodel
    mv pose_iter_440000.caffemodel coco_pose_iter_440000.caffemodel
    # Caffemodel for MPI
    wget http://posefs1.perception.cs.cmu.edu/Users/tsimon/Projects/coco/data/models/mpi/pose_iter_160000.caffemodel
    mv pose_iter_160000.caffemodel mpi_pose_iter_160000.caffemodel

    cd $CWD
}

if [[ "$DIR" != "scanner" ]] && [[ "$1" != "-f" ]];
then
    echo "Warning: you must run this script from the Scanner repo root, and I don't think you are."
    echo "Run this again with -f if you're sure."
else
    prog
fi

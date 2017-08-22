#!/bin/bash

CWD=`pwd`
DIR=`basename $CWD`

prog() {
    mkdir nets/cpm2
    cd nets/cpm2

    # Prototxt for MPI version
    wget https://storage.googleapis.com/scanner-data/nets/cpm2/mpi_pose_deploy_linevec.prototxt
    # Caffemodel for MPI
    wget https://storage.googleapis.com/scanner-data/nets/cpm2/mpi_pose_iter_160000.caffemodel

    # Prototxt for COCO
    wget https://storage.googleapis.com/scanner-data/models/cpm2/coco_pose_deploy_linevec.prototxt
    # Caffemodel for COCO
    https://storage.googleapis.com/scanner-data/models/cpm2/coco_pose_iter_440000.caffemodel

    cd $CWD
}

if [[ "$DIR" != "scanner" ]] && [[ "$1" != "-f" ]];
then
    echo "Warning: you must run this script from the Scanner repo root, and I don't think you are."
    echo "Run this again with -f if you're sure."
else
    prog
fi

#!/bin/bash
set -e

NO_CACHE=false
CORES=$(nproc)

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

for dir in $DIR/*/
do
    base=`basename ${dir%*/}`

    cp $DIR/../deps.sh $dir/deps.sh

    function build {
        TYPE=$1
        TAG=$2
        BASE_TAG=$3

        docker build \
               --build-arg cores=$CORES \
               --build-arg base_tag=$BASE_IMAGE \
               --no-cache=$NO_CACHE \
               -t scannerresearch/scanner-base:$TAG \
               -f $dir/Dockerfile.$TYPE \
               $dir
    }

    function build_chain {
        TYPE=$1
        TAG=$2
        BASE_TAG=$3

        build base $TYPE $TAG-base $BASE_IMAGE
        build $TYPE $TAG $TAG-base
    }

    function push {
        docker push scannerresearch/scanner-base:$1
    }

    function build_push_gpu {
        CUDA_VERSION=$1
        CUDNN_VERSION=$2
        BASE_TAG=nvidia/cuda:${CUDA_VERSION}-{CUDNN_VERSION}-devel-ubuntu16.04
        TAG=$base-gpu-$CUDA_VERSION-$CUDNN_VERSION

        build_chain gpu $TAG $BASE_TAG
        push $TAG
    }


    base_tag=scannerresearch/scanner-base:$base

    # Build cpu with ubuntu:16.04
    build_chain cpu $base-cpu ubuntu:16.04
    push $base-cpu

    # GPU
    build_push_gpu 8.0 cudnn6
    build_push_gpu 8.0 cudnn7
    build_push_gpu 9.1 cudnn7
done

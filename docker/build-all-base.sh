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
        local TYPE=$1
        local TAG=$2
        local BASE_TAG=$3

        docker build \
               --build-arg cores=$CORES \
               --build-arg base_tag=$BASE_TAG \
               --no-cache=$NO_CACHE \
               -t scannerresearch/scanner-base:$TAG \
               -f $dir/Dockerfile.$TYPE \
               $dir
    }

    function build_chain {
        local TYPE=$1
        local TAG=$2
        local BASE_TAG=$3

        build base $TAG-base $BASE_TAG
        build $TYPE $TAG scannerresearch/scanner-base:$TAG-base
    }

    function push {
        docker push scannerresearch/scanner-base:$1
    }

    function build_push_gpu {
        local CUDA_MAJOR_VERSION=$1
        local CUDA_VERSION=$2
        local CUDNN_VERSION=$3
        local BASE_TAG=nvidia/cuda:${CUDA_VERSION}-${CUDNN_VERSION}-devel-ubuntu16.04
        local TAG=$base-gpu-$CUDA_VERSION-$CUDNN_VERSION

        build_chain gpu${CUDA_MAJOR_VERSION} $TAG $BASE_TAG
        push $TAG
    }


    base_tag=scannerresearch/scanner-base:$base

    # Build cpu with ubuntu:16.04
    build_chain cpu $base-cpu ubuntu:16.04
    push $base-cpu

    # GPU
    build_push_gpu 8 8.0 cudnn6
    build_push_gpu 8 8.0 cudnn7
    build_push_gpu 9 9.1 cudnn7
done

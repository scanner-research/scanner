#!/bin/bash

NO_CACHE=false
CORES=16

for dir in ./*/
do
    dir=`basename ${dir%*/}`

    base_tag=scannerresearch/scanner-base:$dir
    docker build --build-arg cores=$CORES \
           --no-cache=$NO_CACHE -t $base_tag \
           -f $dir/Dockerfile.base $dir

    tag=scannerresearch/scanner-base:$dir-gpu
    docker build \
           --build-arg base_tag=$base_tag \
           --build-arg cores=$CORES \
           --no-cache=$NO_CACHE -t $tag \
           -f $dir/Dockerfile.gpu $dir
    docker push $tag

    tag=scannerresearch/scanner-base:$dir-cpu
    docker build \
           --build-arg base_tag=$base_tag \
           --build-arg cores=$CORES \
           --build-arg cpu_only=ON \
           --no-cache=$NO_CACHE -t $tag \
           -f $dir/Dockerfile.cpu $dir
    docker push $tag
done

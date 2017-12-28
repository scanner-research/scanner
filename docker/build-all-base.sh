#!/bin/bash

NO_CACHE=false
CORES=$(nproc)

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

for dir in $DIR/*/
do
    base=`basename ${dir%*/}`

    cp $DIR/../deps.sh $dir/deps.sh

    base_tag=scannerresearch/scanner-base:$base

    function build {
        docker build \
               --build-arg cores=$CORES \
               --build-arg base_tag=$base_tag \
               --no-cache=$NO_CACHE \
               -t scannerresearch/scanner-base:$1 \
               -f $dir/Dockerfile.base \
               .
    }

    function push {
        docker push scannerresearch/scanner-base:$1
    }

    build $base

    build gpu
    push gpu

    build cpu
    push cpu
done

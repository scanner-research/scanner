#!/bin/bash
set -e

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
               -t scannerresearch/scanner-base:$2 \
               -f $dir/Dockerfile.$1 \
               $dir
    }

    function push {
        docker push scannerresearch/scanner-base:$1
    }

    build base $base

    build cpu $base-cpu
    push $base-cpu

    build gpu $base-gpu
    push $base-gpu

done

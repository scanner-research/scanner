#!/bin/bash

# The Travis VM isn't big enough to hold two Docker images of Scanner,
# so we have to push and delete the CPU image before building the GPU one.

set -e

build_docker() {
    # We add -local to make sure it doesn't run the remote image if the build fails.
    if [[ "$1" -eq "cpu" ]]
    then
         docker build -t $DOCKER_REPO:$1-local . --build-arg gpu=OFF
         docker run $DOCKER_REPO:$1-local /bin/bash -c "cd /opt/scanner/build && make test"
    else
         docker build -t $DOCKER_REPO:$1-local . --build-arg gpu=ON
    fi
    docker tag $DOCKER_REPO:$1-local $DOCKER_REPO:$1
    docker push $DOCKER_REPO:$1
    docker rm $(docker ps -a -f status=exited -q)
    docker rmi -f $DOCKER_REPO:$1
}

docker login -e="$DOCKER_EMAIL" -u="$DOCKER_USER" -p="$DOCKER_PASS"
build_docker cpu
build_docker gpu

#!/bin/bash

# The Travis VM isn't big enough to hold two Docker images of Scanner,
# so we have to push and delete the CPU image before building the GPU one.

build_docker() {
    docker build -t $DOCKER_REPO:$1 . --build-arg gpu=$2
    docker run $DOCKER_REPO:$1 /bin/bash -c "cd /opt/scanner/build && make test"
    docker push $DOCKER_REPO:$1
    docker rm $(docker ps -a -f status=exited -q)
    docker rmi -f $DOCKER_REPO:$1
}

docker login -e="$DOCKER_EMAIL" -u="$DOCKER_USER" -p="$DOCKER_PASS"
build_docker cpu OFF
build_docker gpu ON

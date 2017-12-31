#!/bin/bash

set -e

# The Travis VM isn't big enough to hold two Docker images of Scanner,
# so we have to push and delete the CPU image before building the GPU one.

if [ "$TRAVIS_BRANCH" = "master" -a "$TRAVIS_PULL_REQUEST" = "false" ]; then
    PUSH=0
else
    PUSH=1
fi

build_docker() {
    # We add -local to make sure it doesn't run the remote image if the build fails.
    if [ "$1" = "cpu" ]
    then
        docker build -t $DOCKER_REPO:$1-local . \
               --build-arg gpu=OFF --build-arg tag=cpu \
               -f docker/Dockerfile.scanner
        # travis_wait allows tests to run for N minutes with no output
        # https://docs.travis-ci.com/user/common-build-problems/#Build-times-out-because-no-output-was-received
        docker run $DOCKER_REPO:$1-local /bin/bash \
               -c "cd /opt/scanner/build && CTEST_OUTPUT_ON_FAILURE=1 make test"
        docker rm $(docker ps -a -f status=exited -q)
    else
        docker build -t $DOCKER_REPO:$1-local . \
               --build-arg gpu=ON --build-arg tag=gpu \
               -f docker/Dockerfile.scanner
    fi

    if [ $PUSH -eq 0 ]; then
        docker tag $DOCKER_REPO:$1-local $DOCKER_REPO:$1
        docker push $DOCKER_REPO:$1
        docker rmi -f $DOCKER_REPO:$1
    fi

    docker rmi -f $DOCKER_REPO:$1-local
}

if [ $PUSH -eq 0 ]; then
    docker login -u="$DOCKER_USER" -p="$DOCKER_PASS"
fi

build_docker cpu
build_docker gpu

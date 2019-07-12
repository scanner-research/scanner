#!/bin/bash

# Writing output (bell) keeps travis from timing out
# https://github.com/travis-ci/travis-ci/issues/7961
function bell() {
  while true; do
    echo -e "\a"
    sleep 60
  done
}
bell &

set -e

# The Travis VM isn't big enough to hold two Docker images of Scanner,
# so we have to push and delete the CPU image before building the GPU one.

if [[ "$TRAVIS_BRANCH" = "master" ]]; then
    TRAVIS_TAG="latest"
fi

if [[ ("$TRAVIS_BRANCH" = "master" || "$TRAVIS_BRANCH" = "$TRAVIS_TAG") && \
          "$TRAVIS_PULL_REQUEST" = "false" ]]; then
    PUSH=0
else
    PUSH=1
    yes | docker login -u="$DOCKER_USER" -p="$DOCKER_PASS"
fi

build_docker() {
    # We add -local to make sure it doesn't run the remote image if the build fails.
    if [ "$1" = "cpu" ]; then
        docker build -t $DOCKER_REPO:$1-local . \
               --build-arg gpu=OFF --build-arg tag=cpu --build-arg deps_opt='' \
               -f docker/Dockerfile.scanner
    else
        # Parse gpu build type
        local TAG=$1
        docker build -t $DOCKER_REPO:$1-local . \
               --build-arg gpu=ON \
               --build-arg tag=$TAG \
               --build-arg deps_opt='-g' \
               -f docker/Dockerfile.scanner
    fi

    echo "$TRAVIS_BUILD_STAGE_NAME"
    if [ "$TRAVIS_BUILD_STAGE_NAME" = "Test build" ]; then
        docker tag $DOCKER_REPO:$1-local $DOCKER_TEST_REPO:$1-$TRAVIS_BUILD_NUMBER
        docker push $DOCKER_TEST_REPO:$1-$TRAVIS_BUILD_NUMBER
        docker rmi -f $DOCKER_TEST_REPO:$1-$TRAVIS_BUILD_NUMBER
    elif [ $PUSH -eq 0 ]; then
        docker tag $DOCKER_REPO:$1-local $DOCKER_REPO:$1-$TRAVIS_TAG
        docker push $DOCKER_REPO:$1-$TRAVIS_TAG
        docker rmi -f $DOCKER_REPO:$1-$TRAVIS_TAG
    fi
}

build_docker $BUILD_TYPE

exit $?

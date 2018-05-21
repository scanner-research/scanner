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
               --build-arg gpu=OFF --build-arg tag=cpu --build-arg deps_opt='' \
               -f docker/Dockerfile.scanner

        # We run the tests as non-root user because Postgres test uses initdb which requires not being root
        docker run $DOCKER_REPO:$1-local /bin/bash \
               -c "adduser --disabled-password --gecos \"\" user && su -c \"cd /opt/scanner && bash build.sh && cd /opt/scanner/build && CTEST_OUTPUT_ON_FAILURE=1 make test\" user"
        docker rm $(docker ps -a -f status=exited -q)
    else
        # Parse gpu build type
        local TAG=$1
        docker build -t $DOCKER_REPO:$1-local . \
               --build-arg gpu=ON \
               --build-arg tag=$TAG \
               --build-arg deps_opt='-g' \
               -f docker/Dockerfile.scanner
    fi

    if [ $PUSH -eq 0 ]; then
        docker tag $DOCKER_REPO:$1-local $DOCKER_REPO:$1
        docker push $DOCKER_REPO:$1
        docker rmi -f $DOCKER_REPO:$1
    fi
}

if [ $PUSH -eq 0 ]; then
    yes | docker login -u="$DOCKER_USER" -p="$DOCKER_PASS"
fi

build_docker $BUILD_TYPE

exit $?


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

test_docker() {
    INSTALL_SCANNERTOOLS="pushd /tmp && \
      git clone https://github.com/scanner-research/scannertools -b redesign && \
      cd scannertools/scannertools && pip3 install . && popd"

    if [[ "$TEST_TYPE" = "cpp" ]]; then
        TEST_COMMAND="cd /opt/scanner/build && CTEST_OUTPUT_ON_FAILURE=1 make test ARGS='-V -E PythonTests'"
    elif [[ "$TEST_TYPE" = "tutorials" ]]; then
        TEST_COMMAND="$INSTALL_SCANNERTOOLS && cd /opt/scanner/ && python3 setup.py test --addopts '-k test_tutorial'"
    elif [[ "$TEST_TYPE" = "integration" ]]; then
        TEST_COMMAND="cd /opt/scanner/ && python3 setup.py test --addopts '-k \\\"not test_tutorial\\\"'"
    fi
    # We add -local to make sure it doesn't run the remote image if the build fails.
    docker pull $DOCKER_TEST_REPO:$1-$TRAVIS_BUILD_NUMBER
    docker run $DOCKER_TEST_REPO:$1-$TRAVIS_BUILD_NUMBER /bin/bash \
           -c "adduser --disabled-password --gecos \"\" user && (yes | pip3 uninstall grpcio protobuf) && chmod -R 777 /opt/scanner && su -c \"cd /opt/scanner/dist && (yes | pip3 install --user *) && $TEST_COMMAND\" user"
    docker rm $(docker ps -a -f status=exited -q)
    docker rmi -f $DOCKER_REPO:$1-$TRAVIS_BUILD_NUMBER
}

yes | docker login -u="$DOCKER_USER" -p="$DOCKER_PASS"

test_docker $BUILD_TYPE

exit $?

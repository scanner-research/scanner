#!/bin/bash

# docker build -t $DOCKER_REPO:cpu . --build-arg gpu=OFF
# docker run $DOCKER_REPO:cpu /bin/bash -c "cd /opt/scanner/build && make test"
# docker rm $(docker ps -a -f status=exited -q)
# docker build -t $DOCKER_REPO:gpu . --build-arg gpu=ON

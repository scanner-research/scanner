#!/bin/bash

# The Travis VM isn't big enough to hold two Docker images of Scanner,
# so we have to push and delete the CPU image before building the GPU one.

docker build -t $DOCKER_REPO:cpu . --build-arg gpu=OFF
docker run $DOCKER_REPO:cpu /bin/bash -c "cd /opt/scanner/build && make test"
docker login -e="$DOCKER_EMAIL" -u="$DOCKER_USER" -p="$DOCKER_PASS"
docker push $DOCKER_REPO:cpu
docker rm $(docker ps -a -f status=exited -q)
docker rmi -f $DOCKER_REPO:cpu
docker build -t $DOCKER_REPO:gpu . --build-arg gpu=ON
docker push $DOCKER_REPO:gpu

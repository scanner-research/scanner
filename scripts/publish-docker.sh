#!/bin/bash

docker login -e="$DOCKER_EMAIL" -u="$DOCKER_USER" -p="$DOCKER_PASS"
docker push $DOCKER_REPO:cpu
# docker push $DOCKER_REPO:gpu

#!/bin/bash

OPENVINO_REQ_PKGS=(
        libusb-1.0-0-dev
        libgstreamer1.0-0
        gstreamer1.0-plugins-base
        gstreamer1.0-plugins-good
        gstreamer1.0-plugins-bad
    )
apt update
apt install -y ${OPENVINO_REQ_PKGS[@]}

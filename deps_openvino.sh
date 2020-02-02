#!/bin/bash

if [[ "$OSTYPE" == "linux-gnu" ]]; then
    OPENVINO_REQ_PKGS=(
        libusb-1.0-0-dev
        libgstreamer1.0-0
        gstreamer1.0-plugins-base
        gstreamer1.0-plugins-good
        gstreamer1.0-plugins-bad
    )
    apt update
    apt install -y ${OPENVINO_REQ_PKGS[@]}
        # ...
elif [[ "$OSTYPE" == "darwin"* ]]; then
    echo "Support for Scanner with OpenVINO in Mac OSX is not available at this time." 
    exit 1  
        # Mac OSX
else
    # Unknown.
    echo "Unknown OSTYPE: $OSTYPE. Exiting."
    exit 1
fi


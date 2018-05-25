#!/bin/bash

PKG=scannerpy

if [[ "$OSTYPE" == "linux-gnu" ]]; then
    cores=$(nproc)
        # ...
elif [[ "$OSTYPE" == "darwin"* ]]; then
    cores=$(gnproc)
        # Mac OSX
else
    # Unknown.
    echo "Unknown OSTYPE: $OSTYPE. Exiting."
    exit 1
fi

pushd build
if make install -j$cores; then
    popd
    (yes | pip3 uninstall $PKG)
    rm -rf dist && \
        python3 setup.py install;
else
    popd
fi

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
make -j$cores
rc=$?
if [ $rc -eq 0 ]; then
    popd
    if rm -rf dist && \
        python3 setup.py bdist_wheel;
    then
        cwd=$(pwd)
        # cd to /tmp to avoid name clashes with Python module name and any
        # directories of the same name in our cwd
        pushd /tmp
        (yes | pip3 uninstall $PKG)
        (yes | pip3 install $cwd/dist/*)
        popd
    fi
else
    popd
    exit $rc
fi

#!/bin/bash
set -e

PKG=scannerpy
NO_DEPS=false

while [[ $# -gt 0 ]]
do
key="$1"

case $key in
    -n|--no-deps)
        NO_DEPS=true
        shift
        ;;
    *)
esac
done

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
if make -j$cores; then
    popd
    if rm -rf dist && \
        python3 setup.py bdist_wheel;
    then
        cwd=$(pwd)
        # cd to /tmp to avoid name clashes with Python module name and any
        # directories of the same name in our cwd
        pushd /tmp
        (yes | pip3 uninstall $PKG)
        if $NO_DEPS; then
            (yes | pip3 install --user --no-deps $cwd/dist/*);
        else
            (yes | pip3 install --user $cwd/dist/*);
        fi
        popd
    fi
else
    popd
    exit 1
fi

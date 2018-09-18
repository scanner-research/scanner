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

if [[ ("$TRAVIS_BRANCH" = "$TRAVIS_TAG") && "$TRAVIS_PULL_REQUEST" = "false" ]]; then
    PUSH=0
else
    PUSH=1
fi

build_osx() {
    # Need python3
    brew outdated python || brew upgrade python

    # Install Scanner depenendencies
    brew tap scanner-research/homebrew-scanner
    brew install scanner --only-dependencies

    sudo bash ./deps.sh -a -ng \
         --prefix /usr/local \
         --with-ffmpeg /usr/local \
         --with-opencv /usr/local \
         --with-protobuf /usr/local \
         --with-grpc /usr/local \
         --with-caffe /usr/local \
         --with-hwang /usr/local \
         --with-pybind /usr/local \
         --with-libpqxx /usr/local \
         --with-storehouse /usr/local \
         --with-hwang /usr/local

    mkdir -p build
    cd build
    cmake ..
    make -j

    cd ..
    yes | pip3 install grpcio==1.12.0
    bash ./build.sh
    yes | pip3 install grpcio==1.14.0
    yes | pip3 install protobuf==3.6.0

    # Test the build
    yes | python3 -c "import scannerpy; scannerpy.Database()"

    yes | pip3 uninstall scannerpy

    if [ $PUSH -eq 0 ]; then
        git config --global user.name "${COMMIT_USER}"
        git config --global user.email "${COMMIT_EMAIL}"

        # Unencrypt ssh key
        mkdir -p ~/.ssh/
        chmod 0700 ~/.ssh/
        openssl aes-256-cbc -K $encrypted_519f11e8a6d4_key -iv $encrypted_519f11e8a6d4_iv -in .travis/travisci_rsa.enc -out .travis/travisci_rsa -d
        chmod 0600 .travis/travisci_rsa
        cp .travis/travisci_rsa ~/.ssh/id_rsa
        cp .travis/travisci_rsa.pub ~/.ssh/id_rsa.pub
        chmod 0744 ~/.ssh/id_rsa.pub
        ls -lah .travis
        ls -lah ~/.ssh/

        eval `ssh-agent -s`
        ssh-add
        rm -fr ~/.ssh/known_hosts
        ssh-keyscan -t rsa github.com >> ~/.ssh/known_hosts

        # Pull down tar.gz to get sha256
        export URL=https://github.com/scanner-research/scanner/archive/$TRAVIS_TAG.tar.gz
        wget $URL
        export SHA256=$(shasum -a 256 $TRAVIS_TAG.tar.gz | awk '{ print $1 }')

        # Go to scanner homebrew directory to update url and sha256
        cd /usr/local/Homebrew/Library/Taps/scanner-research/homebrew-scanner/Formula
        sed -i "" "s|  url .*|  url \"$URL\"|g" scanner.rb
        sed -i "" "s|  sha256 .*|  sha256 \"$SHA256\"|g" scanner.rb

        # Test new homebrew version

        # Since the previous Scanner installation installs files to /usr/local (such as
        # halide, gtest, gmock headers), the install will return a non-zero exit code
        # indicating it failed to link. We capture that error here and then rely on
        # brew link --overwrite to deal with the linking issue. If the package failed to
        # install, brew link should fail as well.
        brew uninstall --force scanner | true
        brew cleanup --force -s scanner | true
        brew install --verbose --debug scanner | true
        brew link --overwrite scanner | true
        pip3 install scannerpy

        yes | python3 -c "import scannerpy; scannerpy.Database()"

        # Push new homebrew version
        git commit -m "Automated update for Scanner version $TRAVIS_TAG"
        git push
    fi
}

build_osx 

exit $?

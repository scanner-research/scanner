#!/bin/bash

CWD=`pwd`
DIR=`basename $CWD`

prog() {
    mkdir -p features/resnet
    cd features/resnet

    wget https://iuxblw-bn1306.files.1drv.com/y3ml4MHciDBiEDaTSmHlVGB9Hm9cIQNS53sbuCwaolComo2PZ55hhPo5SijUqhtgTv8cad4vvbn7LOY_KPNwJsz-NQTpJENAFTTdVIML1J7-_1uU2hQHE54eak7bf_ZjTJK9aOKxzBPrxrtm8Uu0d3TUPDmcG9ieDoSuonT_YpdKC0/ResNet-50-deploy.prototxt?download&psid=1
    mv ResNet-50-deploy.prototxt?download ResNet-50-deploy.prototxt

    wget https://iuxblw-bn1306.files.1drv.com/y3mUZccSR9x9zlIg_J9kGKeSlmZVvbNxfV8Rajw74tIIsvfHH3AH9GZ3cmYZoaePkTzliM1K5fDEzPKW0z-BLyvxm8rdzNvwgwjUjo2RMsMcxBVd8gzKCKC6WPowuGzRDB9wFK942ZvywiJ12bwiar8OCKy2NlRQbCUw3f_PRaUVc0/ResNet-50-model.caffemodel?download&psid=1
    mv ResNet-50-model.caffemodel?download ResNet-50-model.caffemodel

    cd $CWD
}

if [[ "$DIR" != "scanner" ]] && [[ "$1" != "-f" ]];
then
    echo "Warning: you must run this script from the Scanner repo root, and I don't think you are."
    echo "Run this again with -f if you're sure."
else
    prog
fi

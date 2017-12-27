#!/bin/bash

cores=$(nproc)

LOCAL_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
BUILD_DIR=$LOCAL_DIR/thirdparty/build
DEFAULT_INSTALL_DIR=$LOCAL_DIR/thirdparty/install
FILES_DIR=$LOCAL_DIR/thirdparty/resources

POSITIONAL=()

# Ask if installed
INSTALL_BOOST=true
INSTALL_FFMPEG=true
INSTALL_OPENCV=true
INSTALL_PROTOBUF=true
INSTALL_GRPC=true
INSTALL_CAFFE=true
INSTALL_HALIDE=true

USE_GPU=false

# Assume not installed
INSTALL_HWANG=true
INSTALL_TINYTOML=true
INSTALL_STOREHOUSE=true

INSTALL_PREFIX=$DEFAULT_INSTALL_DIR

INSTALL_ALL=false

while [[ $# -gt 0 ]]
do
key="$1"

case $key in
    -c|--cores)
    cores="$2"
    shift # past arg
    shift # past value
    ;;
    -g|--use-gpu)
    USE_GPU=true
    shift # past arg
    ;;
    -p|--prefix)
    INSTALL_PREFIX="$2"
    shift # past arg
    shift # past value
    ;;
    -a|--install-all)
    INSTALL_ALL=true
    shift # past arg
    ;;
    *)    # unknown option
    POSITIONAL+=("$1") # save it in an array for later
    shift # past argument
    ;;
esac
done

echo "--------------------------------------------------------------"
echo "|          Scanner Dependency Installation Script            |"
echo "--------------------------------------------------------------"
echo "The script will ask if required dependencies are installed and"
echo "then install missing dependencies to "
echo "$INSTALL_PREFIX"
echo "(customized by specifying (--prefix <dir>)"

set -- "${POSITIONAL[@]}" # restore positional parameters

# Directories for installed dependencies
BOOST_DIR=$INSTALL_PREFIX
FFMPEG_DIR=$INSTALL_PREFIX
OPENCV_DIR=$INSTALL_PREFIX
PROTOBUF_DIR=$INSTALL_PREFIX
GRPC_DIR=$INSTALL_PREFIX
CAFFE_DIR=$INSTALL_PREFIX
HALIDE_DIR=$INSTALL_PREFIX
HWANG_DIR=$INSTALL_PREFIX
STOREHOUSE_DIR=$INSTALL_PREFIX
TINYTOML_DIR=$INSTALL_PREFIX

export C_INCLUDE_PATH=$INSTALL_PREFIX/include:$C_INCLUDE_PATH
export LD_LIBRARY_PATH=$INSTALL_PREFIX/lib:$LD_LIBRARY_PATH
export PATH=$INSTALL_PREFIX/bin:$PATH
export PKG_CONFIG_PATH=$INSTALL_PREFIX/lib/pkgconfig:$PGK_CONFIG_PATH

mkdir -p $BUILD_DIR
mkdir -p $INSTALL_PREFIX

if [[ $INSTALL_ALL == false ]]; then
    # Ask about each library
    while true; do
        echo "Do you have boost>=1.63.0 installed with the modules: "
        echo -n "thread, program_options, regex, python, numpy? [y/N]: "
        read yn
        if [[ $yn == y ]] || [[ $yn == Y ]]; then
            INSTALL_BOOST=false
            echo -n "Where is your boost install? [/usr/local]: "
            read install_location
            if [[ $install_location == "" ]]; then
                BOOST_DIR=/usr/local
            else
                BOOST_DIR=$install_location
            fi
            break
        else
            INSTALL_BOOST=true
            break
        fi
    done
    while true; do
        echo -n "Do you have ffmpeg>=3.3.1 installed? [y/N]: "
        read yn
        if [[ $yn == y ]] || [[ $yn == Y ]]; then
            INSTALL_FFMPEG=false
            echo -n "Where is your ffmpeg install? [/usr/local]: "
            read install_location
            if [[ $install_location == "" ]]; then
                FFMPEG_DIR=/usr/local
            else
                FFMPEG_DIR=$install_location
            fi
            break
        else
            INSTALL_FFMPEG=true
            break
        fi
    done
    while true; do
        echo -n "Do you have opencv>=3.2.0 with contrib installed? [y/N]: "
        read yn
        if [[ $yn == y ]] || [[ $yn == Y ]]; then
            INSTALL_OPENCV=false
            echo -n "Where is your opencv install? [/usr/local]: "
            read install_location
            if [[ $install_location == "" ]]; then
                OPENCV_DIR=/usr/local
            else
                OPENCV_DIR=$install_location
            fi
            break
        else
            INSTALL_OPENCV=true
            break
        fi
    done
    while true; do
        echo -n "Do you have protobuf>=3.4.0 installed? [y/N]: "
        read yn
        if [[ $yn == y ]] || [[ $yn == Y ]]; then
            INSTALL_PROTOBUF=false
            echo -n "Where is your protobuf install? [/usr/local]: "
            read install_location
            if [[ $install_location == "" ]]; then
                PROTOBUF_DIR=/usr/local
            else
                PROTOBUF_DIR=$install_location
            fi
            break
        else
            INSTALL_PROTOBUF=true
            break
        fi
    done
    while true; do
        echo -n "Do you have grpc>=1.7.2 installed? [y/N]: "
        read yn
        if [[ $yn == y ]] || [[ $yn == Y ]]; then
            INSTALL_GRPC=false
            echo -n "Where is your grpc install? [/usr/local]: "
            read install_location
            if [[ $install_location == "" ]]; then
                GRPC_DIR=/usr/local
            else
                GRPC_DIR=$install_location
            fi
            break
        else
            INSTALL_GRPC=true
            break
        fi
    done
    while true; do
        echo -n "Do you have halide (release_2016_10_25) installed? [y/N]: "
        read yn
        if [[ $yn == y ]] || [[ $yn == Y ]]; then
            INSTALL_HALIDE=false
            echo -n "Where is your halide install? [/usr/local]: "
            read install_location
            if [[ $install_location == "" ]]; then
                HALIDE_DIR=/usr/local
            else
                HALIDE_DIR=$install_location
            fi
            break
        else
            INSTALL_HALIDE=true
            break
        fi
    done

    while true; do
        echo -n "Do you have caffe>=rc5 or intel-caffe>=1.0.6 installed? [y/N]: "
        read yn
        if [[ $yn == y ]] || [[ $yn == Y ]]; then
            INSTALL_CAFFE=false
            echo -n "Where is your caffe install? [/usr/local]: "
            read install_location
            if [[ $install_location == "" ]]; then
                CAFFE_DIR=/usr/local
            else
                CAFFE_DIR=$install_location
            fi
            break
        else
            INSTALL_CAFFE=true
            echo -n "Do you plan to use GPUs for CNN evaluation? [y/N]: "
            read yn
            if [[ $yn == y ]] || [[ $yn == Y ]]; then
                USE_GPU=true
                break
            else
                USE_GPU=false
                break
            fi
        fi
    done
fi

if [[ $INSTALL_BOOST == true ]] && [[ ! -f $BUILD_DIR/boost.done ]] ; then
    echo "Installing boost 1.63.0..."
    cd $BUILD_DIR
    rm -fr boost*
    wget "https://dl.bintray.com/boostorg/release/1.63.0/source/boost_1_63_0.tar.gz" && \
        tar -xf boost_1_63_0.tar.gz && cd boost_1_63_0 && ./bootstrap.sh && \
        ./b2 install --prefix=$INSTALL_PREFIX -j${cores} && \
        rm -rf $BUILD_DIR/boost_1_63_0.tar.gz && touch $BUILD_DIR/boost.done \
            || { echo 'Installing boost failed!' ; exit 1; }
    echo "Done installing boost 1.63.0"
fi


if [[ $INSTALL_FFMPEG == true ]] && [[ ! -f $BUILD_DIR/ffmpeg.done ]] ; then
    echo "Installing ffmpeg 3.3.1..."
    # FFMPEG
    cd $BUILD_DIR
    rm -fr ffmpeg
    git clone -b n3.3.1 https://git.ffmpeg.org/ffmpeg.git && cd ffmpeg && \
    ./configure --prefix=$INSTALL_PREFIX --extra-version=0ubuntu0.16.04.1 \
                --toolchain=hardened --cc=cc --cxx=g++ --enable-gpl \
                --enable-shared --disable-stripping \
                --disable-decoder=libschroedinger \
                --enable-avresample --enable-libx264 --enable-nonfree && \
    make install -j${cores} && touch $BUILD_DIR/ffmpeg.done \
        || { echo 'Installing ffmpeg failed!' ; exit 1; }
    echo "Done installing ffmpeg 3.3.1"
fi

if [[ $INSTALL_OPENCV == true ]] && [[ ! -f $BUILD_DIR/opencv.done ]]; then
    # OpenCV 3.2.0 + OpenCV contrib
    echo "Installing OpenCV 3.2.0..."
    cd $BUILD_DIR
    rm -rf opencv opencv_contrib ceres-solver
    git clone -b 3.2.0 https://github.com/opencv/opencv && \
        git clone -b 3.2.0  https://github.com/opencv/opencv_contrib && \
        git clone -b 1.12.0 https://github.com/ceres-solver/ceres-solver && \
        cd ceres-solver && mkdir build && cd build && \
        cmake .. -DCMAKE_INSTALL_PREFIX=$INSTALL_PREFIX && \
        make install -j$cores && \
        mkdir -p $BUILD_DIR/opencv/build && cd $BUILD_DIR/opencv/build && \
        cmake -D CMAKE_BUILD_TYPE=Release -D CMAKE_INSTALL_PREFIX=$INSTALL_PREFIX \
              -D BUILD_TESTS=OFF -D BUILD_PERF_TESTS=OFF -D ENABLE_FAST_MATH=1 \
              -D CUDA_FAST_MATH=1 -D WITH_CUBLAS=1 -D WITH_NVCUVID=1 \
              -D BUILD_opencv_rgbd=OFF \
              -D BUILD_opencv_cnn_3dobj=OFF \
              -D BUILD_TIFF=OFF \
              -D WITH_TIFF=OFF \
              -D OPENCV_EXTRA_MODULES_PATH=$BUILD_DIR/opencv_contrib/modules \
              .. && \
        make install -j$cores && touch $BUILD_DIR/opencv.done \
            || { echo 'Installing OpenCV failed!' ; exit 1; }
    echo "Done installing OpenCV 3.2.0"
fi

if [[ $INSTALL_PROTOBUF == true ]] && [[ ! -f $BUILD_DIR/protobuf.done ]] ; then
    # protobuf 3.4.1
    echo "Installing protobuf 3.4.1..."
    cd $BUILD_DIR
    rm -fr protobuf
    git clone -b v3.4.1 https://github.com/google/protobuf.git && \
        cd protobuf && bash ./autogen.sh && \
        ./configure --prefix=$INSTALL_PREFIX && make -j$cores && \
        make install && touch $BUILD_DIR/protobuf.done \
            || { echo 'Installing protobuf failed!' ; exit 1; }
    echo "Done installing protobuf 3.4.1"
fi

if [[ $INSTALL_GRPC == true ]] && [[ ! -f $BUILD_DIR/grpc.done ]] ; then
    # gRPC 1.7.2
    echo "Installing gRPC 1.7.2..."
    cd $BUILD_DIR
    rm -fr grpc
    git clone -b v1.7.2 https://github.com/grpc/grpc && \
        cd grpc && git submodule update --init --recursive && \
        make EXTRA_CPPFLAGS=-I$INSTALL_PREFIX/include \
             EXTRA_LDFLAGS=-L$INSTALL_PREFIX/lib -j$cores && \
        make install EXTRA_CPPFLAGS=-I$INSTALL_PREFIX/include \
             EXTRA_LDFLAGS=-L$INSTALL_PREFIX/lib prefix=$INSTALL_PREFIX && \
        ldconfig -n $INSTALL_PREFIX/lib && \
        touch $BUILD_DIR/grpc.done \
            || { echo 'Installing gRPC failed!' ; exit 1; }
    echo "Done installing gRPC 1.7.2"
fi

if [[ $INSTALL_HALIDE == true ]] && [[ ! -f $BUILD_DIR/halide.done ]] ; then
    # Halide
    echo "Installing Halide..."
    cd $BUILD_DIR
    rm -fr Halide
    git clone -b release_2016_10_25 https://github.com/halide/Halide && \
        cd Halide && \
        make distrib -j$cores && \
        cp -r distrib/* $INSTALL_PREFIX && \
        touch $BUILD_DIR/halide.done \
            || { echo 'Installing Halide failed!' ; exit 1; }
    echo "Done installing Halide"
fi

if [[ $INSTALL_STOREHOUSE == true ]] && [[ ! -f $BUILD_DIR/storehouse.done ]] ; then
    echo "Installing storehouse..."
    cd $BUILD_DIR
    rm -fr storehouse
    git clone https://github.com/scanner-research/storehouse && \
        cd storehouse && git checkout b3fde18c8a62b4249a4fbd5ea5f893ecd0d246fb && \
        cd thirdparty && mkdir build && cd build && \
        cmake .. -DCMAKE_INSTALL_PREFIX=$INSTALL_PREFIX && \
        make -j${cores} && cd ../../ && \
        mkdir build && cd build && \
        cmake .. -DCMAKE_INSTALL_PREFIX=$INSTALL_PREFIX -DBOOST_ROOT=$BOOST_DIR && make -j${cores} && \
        make install && \
        touch $BUILD_DIR/storehouse.done \
            || { echo 'Installing storehouse failed!' ; exit 1; }
    echo "Done installing storehouse"
fi

if [[ $INSTALL_HWANG == true ]] && [[ ! -f $BUILD_DIR/hwang.done ]] ; then
    echo "Installing hwang..."
    cd $BUILD_DIR
    rm -fr hwang
    git clone https://github.com/scanner-research/hwang && \
        cd hwang && \
        git checkout 69f04c4e5df9c6db701a0cb5b79f964e66637f87 && \
        bash ./deps.sh -a \
             --with-boost $INSTALL_PREFIX \
             --with-ffmpeg $INSTALL_PREFIX \
             --with-protobuf $INSTALL_PREFIX \
             --cores ${cores} && \
        mkdir -p build && cd build && \
        cmake .. -DCMAKE_INSTALL_PREFIX=$INSTALL_PREFIX -DBUILD_CUDA=ON && \
        make -j${cores} && make install -j${cores} && cd .. && \
        cd python && python setup.py bdist_wheel && \
        pip install dist/hwang-0.1.0-py2-none-any.whl && \
        touch $BUILD_DIR/hwang.done \
            || { echo 'Installing hwang failed!' ; exit 1; }
    echo "Done installing hwang"
fi

if [[ $INSTALL_TINYTOML == true ]]; then
    echo "Installing tinytoml..."
    cd $BUILD_DIR
    rm -fr tinytoml
    git clone https://github.com/mayah/tinytoml.git && \
        cd tinytoml && git checkout 3559856002eee57693349b8a2d8a0cf6250d269c && \
        cp -r include/* $INSTALL_PREFIX/include && \
        touch $BUILD_DIR/tinytoml.done \
            || { echo 'Installing tinytoml failed!' ; exit 1; }
    echo "Done installing tinytoml"
fi

echo "Installing googletest..."
cd $BUILD_DIR
rm -fr googletest
git clone https://github.com/google/googletest && \
    cd googletest && mkdir build && cd build && \
    cmake .. -DCMAKE_INSTALL_PREFIX=$INSTALL_PREFIX && \
    make -j${cores} && make install && \
    touch $BUILD_DIR/googletest.done \
        || { echo 'Installing googletest failed!' ; exit 1; }
echo "Done installing googletest"

if [[ $INSTALL_CAFFE == true ]] && [[ $USE_GPU == false ]] && \
       [[ ! -f $BUILD_DIR/caffe.done ]]; then
    # Intel Caffe 1.0.6
    cd $BUILD_DIR
    rm -fr caffe
    # Use more recent mkldnn commit to fix gcc bug
    git clone -b 1.0.6 https://github.com/intel/caffe && \
        cd caffe && \
        cp $FILES_DIR/caffe/Makefile.config Makefile.config && \
        rm mkldnn.commit && \
        echo "2604f435da7bb9f1896ae37200d91734adfdba9c" > mkldnn.commit && \
        mkdir build && cd build && \
        cmake -DCMAKE_INSTALL_PREFIX=$INSTALL_PREFIX \
              -DCPU_ONLY=ON \
              -DOpenCV_DIR=$INSTALL_PREFIX \
              -DBLAS=mkl \
              .. && \
        make -j${cores} && \
        make install && \
        cd .. && \
        cp -r external/mkl/mklml_lnx_2018.0.20170908/* $INSTALL_PREFIX && \
        cp -r external/mkldnn/install/* $INSTALL_PREFIX && \
        touch $BUILD_DIR/caffe.done \
            || { echo 'Installing caffe failed!' ; exit 1; }
fi

if [[ $INSTALL_CAFFE == true ]] && [[ $USE_GPU == true ]] && \
       [[ ! -f $BUILD_DIR/caffe.done ]]; then
    cd $BUILD_DIR
    # Intel MKL
    rm -fr mkl
    mkdir mkl && \
        cd mkl && \
        wget http://registrationcenter-download.intel.com/akdlm/irc_nas/tec/12414/l_mkl_2018.1.163.tgz && \
        tar -zxf l_mkl_2018.1.163.tgz && \
        cp $FILES_DIR/mkl/silent.cfg silent.cfg && \
        echo "PSET_INSTALL_DIR=$INSTALL_PREFIX/intel" >> silent.cfg && \
        cd l_mkl_2018.1.163 && \
        bash install.sh --cli-mode --silent ../silent.cfg

    cd $BUILD_DIR
    # Caffe rc5
    rm -fr caffe
    git clone -b rc5 https://github.com/BVLC/caffe && \
        cd caffe && cp $FILES_DIR/caffe/Makefile.config Makefile.config && \
        mkdir build && cd build && \
        cmake -DCMAKE_INSTALL_PREFIX=$INSTALL_PREFIX \
              -DCMAKE_PREFIX_PATH=$INSTALL_PREFIX \
              -DINTEL_ROOT=$INSTALL_PREFIX/intel \
              -DBLAS=mkl \
              -DCUDA_ARCH_NAME="Manual" \
              -DCUDA_ARCH_BIN="30 35 50 60 61" \
              -DCUDA_ARCH_PTX="30 35 50 60 61" \
              -DOpenCV_DIR=$INSTALL_PREFIX \
              .. && \
        make -j${cores} && \
        make install && \
        touch $BUILD_DIR/caffe.done \
            || { echo 'Installing caffe failed!' ; exit 1; }
fi

DEP_FILE=$LOCAL_DIR/dependencies.txt
rm -f $DEP_FILE
echo "BOOST_DIR=$BOOST_DIR" >> $DEP_FILE
echo "FFMPEG_DIR=$FFMPEG_DIR" >> $DEP_FILE
echo "OpenCV_DIR=$OPENCV_DIR" >> $DEP_FILE
echo "PROTOBUF_DIR=$PROTOBUF_DIR" >> $DEP_FILE
echo "GRPC_DIR=$GRPC_DIR" >> $DEP_FILE
echo "CAFFE_DIR=$CAFFE_DIR" >> $DEP_FILE
echo "Halide_DIR=$HALIDE_DIR" >> $DEP_FILE
echo "Hwang_DIR=$HWANG_DIR" >> $DEP_FILE
echo "STOREHOUSE_DIR=$STOREHOUSE_DIR" >> $DEP_FILE
echo "TinyToml_DIR=$TINYTOML_DIR" >> $DEP_FILE

echo "Done installing required dependencies!"
echo -n "Add $INSTALL_PREFIX/lib to your LD_LIBRARY_PATH and "
echo -n "add $INSTALL_PREFIX/bin to your PATH so the installed "
echo -n "dependencies can be found! "
echo "e.g. export LD_LIBRARY_PATH=$INSTALL_PREFIX/lib:\$LD_LIBRARY_PATH"
if [[ $INSTALL_OPENCV == true ]]; then
    echo "Add $INSTALL_PREFIX/lib/python2.7/dist-packages to your PYTHONPATH to use OpenCV from Python"
fi
if [[ $INSTALL_CAFFE_CPU == true ]] || [[ $INSTALL_CAFFE_GPU == true ]]; then
    echo "Add $INSTALL_PREFIX/python to your PYTHONPATH to use Caffe from Python"
fi
#echo "Add $INSTALL_PREFIX/lib to your LD_LIBRARY_PATH"

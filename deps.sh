#!/bin/bash

if [[ "$OSTYPE" == "linux-gnu" ]]; then
    cores=$(nproc)
        # ...
elif [[ "$OSTYPE" == "darwin"* ]]; then
    cores=$(sysctl -n hw.ncpu)
        # Mac OSX
else
    # Unknown.
    echo "Unknown OSTYPE: $OSTYPE. Exiting."
    exit 1
fi

LOCAL_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
BUILD_DIR=$LOCAL_DIR/thirdparty/build
DEFAULT_INSTALL_DIR=$LOCAL_DIR/thirdparty/install
FILES_DIR=$LOCAL_DIR/thirdparty/resources
PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')

POSITIONAL=()

# Required, ask if installed
INSTALL_OPENCV=true
INSTALL_PROTOBUF=true
INSTALL_GRPC=true

# Required, and assume not installed
INSTALL_GOOGLETEST=true
INSTALL_TINYTOML=true
INSTALL_STOREHOUSE=true
INSTALL_PYBIND=true

# Optional, ask if needed
NO_FFMPEG=false
INSTALL_FFMPEG=true
NO_OPENPOSE=false
INSTALL_OPENPOSE=true
NO_HALIDE=false
INSTALL_HALIDE=true
NO_CAFFE=false
INSTALL_CAFFE=true
NO_HWANG=false
INSTALL_HWANG=true
NO_OPENVINO=false
INSTALL_OPENVINO=true

# Optional, and assume not installed
NO_LIBPQXX=false
INSTALL_LIBPQXX=true


USE_GPU=false
NO_USE_GPU=false

INSTALL_PREFIX=$DEFAULT_INSTALL_DIR

INSTALL_ALL=false
INSTALL_NONE=false
ROOT_INSTALL=false

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
    -ng|--no-use-gpu)
        NO_USE_GPU=true
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
    -n|--install-none)
        INSTALL_NONE=true
        shift # past arg
        ;;
    --root-install)
        ROOT_INSTALL=true
        shift # past arg
        ;;
    --with-opencv)
        WITH_OPENCV="$2"
        shift # past arg
        shift # past value
        ;;
    --with-protobuf)
        WITH_PROTOBUF="$2"
        shift # past arg
        shift # past value
        ;;
    --with-grpc)
        WITH_GRPC="$2"
        shift # past arg
        shift # past value
        ;;
    --with-storehouse)
        WITH_STOREHOUSE="$2"
        shift # past arg
        shift # past value
        ;;
    --with-pybind)
        WITH_PYBIND="$2"
        shift # past arg
        shift # past value
        ;;
    --without-ffmpeg)
        NO_FFMPEG=true
        shift # past arg
        ;;
    --with-ffmpeg)
        WITH_FFMPEG="$2"
        shift # past arg
        shift # past value
        ;;
    --without-caffe)
        NO_CAFFE=true
        shift # past arg
        ;;
    --with-caffe)
        WITH_CAFFE="$2"
        shift # past arg
        shift # past value
        ;;
    --without-halide)
        NO_HALIDE=true
        shift # past arg
        ;;
    --with-halide)
        WITH_HALIDE="$2"
        shift # past arg
        shift # past value
        ;;
    --without-openpose)
        NO_OPENPOSE=true
        shift # past arg
        ;;
    --with-openpose)
        WITH_OPENPOSE="$2"
        shift # past arg
        shift # past value
        ;;
    --without-libpqxx)
        NO_LIBPQXX=true
        shift # past arg
        ;;
    --with-libpqxx)
        WITH_LIBPQXX="$2"
        shift # past arg
        shift # past value
        ;;
    --without-hwang)
        NO_HWANG=true
        shift # past arg
        ;;
    --with-hwang)
        WITH_HWANG="$2"
        shift # past arg
        shift # past value
        ;;
    --with-openvino)
        WITH_OPENVINO="$2"
        shift # past arg
        shift # past value
        ;;
    --without-openvino)
        NO_OPENVINO=true
        INSTALL_OPENVINO=false
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
echo "(customized by specifying (--prefix <dir>)."

set -- "${POSITIONAL[@]}" # restore positional parameters

if command -v brew >/dev/null 2>&1; then
    HAS_BREW=true
else
    HAS_BREW=false
fi

if command -v conda list >/dev/null 2>&1; then
    # Anaconda is installed, so add lib to prefix path for OpenCV to find
    # PythonLib
    echo "Detected Anaconda, adding lib path to OpenCV and Caffe build"
    py_path=$(dirname $(which python))/../lib
    PY_EXTRA_CMDS="$py_path"
else
    PY_EXTRA_CMDS=""
fi

# Check if we have GPUs by looking for nvidia-smi
BUILD_CV_DNN=false
if command -v nvidia-smi >/dev/null 2>&1; then
    HAVE_GPU=true
    CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $5}' |  sed 's/.$//')
    if [ $CUDA_VERSION -ge '10.0' ]; then
        BUILD_CV_DNN=true
    fi
else
    HAVE_GPU=false
fi


# Force building with GPU when specified
if [[ $USE_GPU == true ]]; then
    HAVE_GPU=true
fi

# Force NOT building with GPU when specified, overriding other commands
if [[ $NO_USE_GPU == true ]]; then
    HAVE_GPU=false
fi

echo ""
echo "Configuration:"
echo "--------------------------------------------------------------"
echo "Detected Python version: $PYTHON_VERSION"
echo "GPUs available:          $HAVE_GPU ($CUDA_VERSION)"
echo ""

# Directories for installed dependencies
FFMPEG_DIR=$INSTALL_PREFIX
OPENCV_DIR=$INSTALL_PREFIX
PROTOBUF_DIR=$INSTALL_PREFIX
GRPC_DIR=$INSTALL_PREFIX
CAFFE_DIR=$INSTALL_PREFIX
HALIDE_DIR=$INSTALL_PREFIX
PYBIND_DIR=$INSTALL_PREFIX
HWANG_DIR=$INSTALL_PREFIX
STOREHOUSE_DIR=$INSTALL_PREFIX
TINYTOML_DIR=$INSTALL_PREFIX
OPENPOSE_DIR=$INSTALL_PREFIX
LIBPQXX_DIR=$INSTALL_PREFIX
OPENVINO_DIR=$INSTALL_PREFIX

if [[ ! -z ${WITH_FFMPEG+x} ]]; then
    INSTALL_FFMPEG=false
    FFMPEG_DIR=$WITH_FFMPEG
fi
if [[ ! -z ${WITH_OPENCV+x} ]]; then
    INSTALL_OPENCV=false
    OPENCV_DIR=$WITH_OPENCV
fi
if [[ ! -z ${WITH_PROTOBUF+x} ]]; then
    INSTALL_PROTOBUF=false
    PROTOBUF_DIR=$WITH_PROTOBUF
fi
if [[ ! -z ${WITH_GRPC+x} ]]; then
    INSTALL_GRPC=false
    GRPC_DIR=$WITH_GRPC
fi
if [[ ! -z ${WITH_CAFFE+x} ]]; then
    INSTALL_CAFFE=false
    CAFFE_DIR=$WITH_CAFFE
fi
if [[ ! -z ${WITH_HALIDE+x} ]]; then
    INSTALL_HALIDE=false
    HALIDE_DIR=$WITH_HALIDE
fi
if [[ ! -z ${WITH_PYBIND+x} ]]; then
    INSTALL_PYBIND=false
    PYBIND_DIR=$WITH_PYBIND
fi
if [[ ! -z ${WITH_HWANG+x} ]]; then
    INSTALL_HWANG=false
    HWANG_DIR=$WITH_HWANG
fi
if [[ ! -z ${WITH_STOREHOUSE+x} ]]; then
    INSTALL_STOREHOUSE=false
    STOREHOUSE_DIR=$WITH_STOREHOUSE
fi
if [[ ! -z ${WITH_OPENPOSE+x} ]]; then
    INSTALL_OPENPOSE=false
    OPENPOSE_DIR=$WITH_OPENPOSE
fi
if [[ ! -z ${WITH_LIBPQXX+x} ]]; then
    INSTALL_LIBPQXX=false
    LIBPQXX_DIR=$WITH_LIBPQXX
fi
if [[ ! -z ${WITH_OPENVINO+x} ]]; then
    INSTALL_OPENVINO=false
    OPENVINO_DIR=$WITH_OPENVINO
fi

export C_INCLUDE_PATH=$INSTALL_PREFIX/include:$C_INCLUDE_PATH
export LD_LIBRARY_PATH=$INSTALL_PREFIX/lib:$LD_LIBRARY_PATH
export PATH=$INSTALL_PREFIX/bin:$PATH
export PKG_CONFIG_PATH=$INSTALL_PREFIX/lib/pkgconfig:$PGK_CONFIG_PATH

mkdir -p $BUILD_DIR
mkdir -p $INSTALL_PREFIX

if [[ $INSTALL_NONE == true ]]; then
    INSTALL_FFMPEG=false
    INSTALL_OPENCV=false
    INSTALL_PROTOBUF=false
    INSTALL_GRPC=false
    INSTALL_CAFFE=false
    INSTALL_HALIDE=false
    INSTALL_OPENPOSE=false
    INSTALL_GOOGLETEST=false
    INSTALL_HWANG=false
    INSTALL_TINYTOML=false
    INSTALL_STOREHOUSE=false
    INSTALL_PYBIND=false
    INSTALL_LIBPQXX=false
    INSTALL_OPENVINO=false

elif [[ $INSTALL_ALL == false ]]; then
    # Ask about each library

    echo "Required dependencies: "
    if [[ -z ${WITH_OPENCV+x} ]]; then
        if [[ $HAS_BREW == true ]] && brew ls --versions opencv > /dev/null; then
            # The package is installed via brew
            INSTALL_OPENCV=false
            OPENCV_DIR=/usr/local
        else
            # The package is not installed via brew
            echo -n "Do you have opencv>=4.2.0 with contrib installed? [y/N]: "
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
            else
                INSTALL_OPENCV=true
                echo "opencv 4.2.0 will be installed at ${OPENCV_DIR}."
            fi
        fi
    fi

    if [[ -z ${WITH_PROTOBUF+x} ]]; then
        if [[ $HAS_BREW == true ]] && brew ls --versions protobuf > /dev/null; then
            # The package is installed via brew
            INSTALL_PROTOBUF=false
            PROTOBUF_DIR=/usr/local
        else
            echo -n "Do you have protobuf>=3.6.1 installed? [y/N]: "
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
            else
                INSTALL_PROTOBUF=true
                echo "protobuf 3.6.1 will be installed at ${PROTOBUF_DIR}."
            fi
        fi
    fi

    if [[ -z ${WITH_GRPC+x} ]]; then
        if [[ $HAS_BREW == true ]] && brew ls --versions grpc > /dev/null; then
            # The package is installed via brew
            INSTALL_GRPC=false
            GRPC_DIR=/usr/local
        else
            echo -n "Do you have grpc==1.16.0 installed? [y/N]: "
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
            else
                INSTALL_GRPC=true
                echo "grpc 1.16.0 will be installed at ${GRPC_DIR}."
            fi
        fi
    fi

    echo "Optional dependencies: "
    if [[ -z ${WITH_OPENVINO+x} ]] && [[ ${NO_OPENVINO+x} != true ]] && [[ "$OSTYPE" == "linux-gnu" ]]; then
        echo -n "Do you need support for OpenVINO Inference Engine? [y/N]: "
        read yn
        if [[ $yn != n ]] && [[ $yn != N ]]; then
            echo -n "Do you have OpenVINO Inference Engine R3.1 2019 installed? [y/N]: "
            read yn
            if [[ $yn == y ]] || [[ $yn == Y ]]; then
                INSTALL_OPENVINO=false
                echo -n "Where is your OpenVINO Inference Engine install? [/usr/local]: "
                read install_location
                if [[ $install_location == "" ]]; then
                    OPENVINO_DIR=/usr/local
                else
                    OPENVINO_DIR=$install_location
                fi
                INTEL_OPENVINO_DIR=$install_location/intel/openvino_2019.3.376
                export INTEL_OPENVINO_DIR=$INTEL_OPENVINO_DIR
                export INTEL_CVSDK_DIR=$INTEL_OPENVINO_DIR
                export InferenceEngine_DIR=$INTEL_OPENVINO_DIR/deployment_tools/inference_engine/share
                export IE_PLUGINS_PATH=$INTEL_OPENVINO_DIR/deployment_tools/inference_engine/lib/intel64
                export HDDL_INSTALL_DIR=$INTEL_OPENVINO_DIR/deployment_tools/inference_engine/external/hddl
                export LD_LIBRARY_PATH=$HDDL_INSTALL_DIR/lib:$INTEL_OPENVINO_DIR/deployment_tools/inference_engine/external/gna/lib:$INTEL_OPENVINO_DIR/deployment_tools/inference_engine/external/mkltiny_lnx/lib:$INTEL_OPENVINO_DIR/deployment_tools/inference_engine/external/tbb/lib:$IE_PLUGINS_PATH:$LD_LIBRARY_PATH
            else
                INSTALL_OPENVINO=true
                echo "OpenVINO Inference Engine 2019 R3.1 will be installed at ${OPENVINO_DIR}."
            fi
        else
            INSTALL_OPENVINO=false
            NO_OPENVINO=true
        fi
    fi

    if [[ -z ${WITH_FFMPEG+x} ]] && [[ ${NO_FFMPEG+x} != true ]]; then
        echo -n "Do you need support for processing video files (e.g. mp4)? [Y/n]: "
        read yn
        if [[ $yn != n ]] && [[ $yn != N ]]; then
            if [[ $HAS_BREW == true ]] && brew ls --versions ffmpeg > /dev/null; then
                # The package is installed via brew
                INSTALL_FFMPEG=false
                FFMPEG_DIR=/usr/local
            else
                echo -n "Do you have ffmpeg>=4.2 installed? [y/N]: "
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
                else
                    INSTALL_FFMPEG=true
                    echo "ffmpeg 4.2 will be installed at ${FFMPEG_DIR}."
                fi
            fi
        else
            INSTALL_FFMPEG=false
            NO_FFMPEG=true
        fi
    fi

    if [[ -z ${WITH_HALIDE+x} ]] && [[ ${NO_HALIDE+x} != true ]]; then
        echo -n "Do you need support for halide pipelines? [y/N]: "
        read yn
        if [[ $yn == y ]] || [[ $yn == Y ]]; then
            echo -n "Do you have halide (release_2018_02_15) installed? [y/N]: "
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
            else
                INSTALL_HALIDE=true
                echo "halide will be installed at ${HALIDE_DIR}."
            fi
        else
            INSTALL_HALIDE=false
            NO_HALIDE=true
        fi
    fi

    if [[ -z ${WITH_OPENPOSE+x} ]] && [[ ${NO_OPENPOSE+x} != true ]]; then
        echo -n "Do you need support for Pose Detection (openpose)? [Y/n]: "
        read yn
        if [[ $yn != n ]] && [[ $yn != N ]]; then
            echo -n "Do you have OpenPose (v1.4.0) installed? [y/N]: "
            read yn
            if [[ $yn == y ]] || [[ $yn == Y ]]; then
                INSTALL_OPENPOSE=false
                echo -n "Where is your OpenPose install? [/usr/local]: "
                read install_location
                if [[ $install_location == "" ]]; then
                    OPENPOSE_DIR=/usr/local
                else
                    OPENPOSE_DIR=$install_location
                fi
            else
                INSTALL_OPENPOSE=true
                echo "openpose 1.4.0 will be installed at ${OPENPOSE_DIR}."
            fi
        else
            INSTALL_OPENPOSE=false
            NO_OPENPOSE=true
        fi
    fi

    if [[ -z ${WITH_CAFFE+x} ]] && [[ ${NO_CAFFE+x} != true ]]; then
        echo -n "Do you need support for Caffe operations? [Y/n]: "
        read yn
        if [[ $yn != n ]] && [[ $yn != N ]]; then
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
            else
                INSTALL_CAFFE=true
                if [[ $HAVE_GPU == true ]]; then
                    echo -n "Do you plan to use GPUs for CNN evaluation? [Y/n]: "
                    read yn
                    if [[ $yn == n ]] || [[ $yn == N ]]; then
                        USE_GPU=false
                    else
                        USE_GPU=true
                    fi
                else
                    USE_GPU=false
                fi
                echo "caffe will be installed at ${CAFFE_DIR}."
            fi
        else
            INSTALL_CAFFE=false
            NO_CAFFE=true
        fi
    fi

    if [[ -z ${WITH_LIBPQXX+x} ]] && [[ ${NO_LIBPQXX+x} != true ]]; then
        echo -n "Do you need support for SQL input/output? [y/N]: "
        read yn
        if [[ $yn == y ]] || [[ $yn == Y ]]; then
            INSTALL_LIBPQXX=true
            echo "libpqxx will be installed at ${LIBPQXX_DIR}."
        else
            INSTALL_LIBPQXX=false
            NO_LIBPQXX=true
        fi
    fi

    if [[ ${NO_HWANG+x} == true ]]; then
        INSTALL_HWANG=false
        NO_HWANG=true
    fi
fi

if [[ $INSTALL_PROTOBUF == true ]] && [[ ! -f $BUILD_DIR/protobuf.done ]] ; then
    # protobuf 3.6.1
    echo "Installing protobuf 3.6.1..."
    cd $BUILD_DIR
    rm -fr protobuf
    git clone -b v3.6.1 https://github.com/google/protobuf.git --depth 1 && \
        cd protobuf && bash ./autogen.sh && \
        ./configure --prefix=$INSTALL_PREFIX && make -j$cores && \
        make install && touch $BUILD_DIR/protobuf.done \
            || { echo 'Installing protobuf failed!' ; exit 1; }
    echo "Done installing protobuf 3.6.1"
fi

if [[ $INSTALL_OPENVINO == true ]] && [[ ! -f $BUILD_DIR/openvino.done ]] ; then
    echo "Installing OpenVINO Inference Engine 2019 R3"
    cd $BUILD_DIR
#    rm -fr openvino
    wget -c http://registrationcenter-download.intel.com/akdlm/irc_nas/16057/l_openvino_toolkit_p_2019.3.376.tgz
    tar xf l_openvino_toolkit*.tgz
    cd l_openvino_toolkit*
    echo "COMPONENTS=intel-openvino-ie-rt-cpu-ubuntu-xenial__x86_64;intel-openvino-ie-rt-gpu-ubuntu-xenial__x86_64;intel-openvino-ie-rt-vpu-ubuntu-xenial__x86_64;intel-openvino-ie-rt-gna-ubuntu-xenial__x86_64;intel-openvino-ie-rt-hddl-ubuntu-xenial__x86_64;intel-openvino-ie-sdk-ubuntu-xenial__x86_64;intel-openvino-model-optimizer__x86_64;intel-openvino-omz-dev__x86_64" >> silent.cfg
    echo "ACCEPT_EULA=accept" > silent.cfg
    echo "CONTINUE_WITH_OPTIONAL_ERROR=yes" >> silent.cfg
    echo "PSET_INSTALL_DIR=${INSTALL_PREFIX}/intel" >> silent.cfg
    echo "CONTINUE_WITH_INSTALLDIR_OVERWRITE=yes" >> silent.cfg
    echo "PSET_MODE=install" >> silent.cfg
    echo "INTEL_SW_IMPROVEMENT_PROGRAM_CONSENT=no" >> silent.cfg
    echo "SIGNING_ENABLED=no" >> silent.cfg
    ./install.sh --ignore-signature --cli-mode --silent silent.cfg || { echo 'Installing OpenVINO failed!' ; exit 1; }
    cd .. && rm l_openvino_toolkit_p_2019.3.376.tgz
    touch $BUILD_DIR/openvino.done
    echo "Done installing OpenVINO Inference Engine 2019 R3.1"
    #This will be needed by OpenCV
    INTEL_OPENVINO_DIR=$INSTALL_PREFIX/intel/openvino_2019.3.376
    export INTEL_OPENVINO_DIR=$INTEL_OPENVINO_DIR
    export INTEL_CVSDK_DIR=$INTEL_OPENVINO_DIR
    export InferenceEngine_DIR=$INTEL_OPENVINO_DIR/deployment_tools/inference_engine/share
    export IE_PLUGINS_PATH=$INTEL_OPENVINO_DIR/deployment_tools/inference_engine/lib/intel64
    export HDDL_INSTALL_DIR=$INTEL_OPENVINO_DIR/deployment_tools/inference_engine/external/hddl
    HDDL_INSTALL_DIR=$INTEL_OPENVINO_DIR/deployment_tools/inference_engine/external/hddl
    IE_PLUGINS_PATH=$INTEL_OPENVINO_DIR/deployment_tools/inference_engine/lib/intel64
    export LD_LIBRARY_PATH=$HDDL_INSTALL_DIR/lib:$INTEL_OPENVINO_DIR/deployment_tools/inference_engine/external/gna/lib:$INTEL_OPENVINO_DIR/deployment_tools/inference_engine/external/mkltiny_lnx/lib:$INTEL_OPENVINO_DIR/deployment_tools/inference_engine/external/tbb/lib:$IE_PLUGINS_PATH:$LD_LIBRARY_PATH
fi

if [[ $INSTALL_OPENCV == true ]] && [[ ! -f $BUILD_DIR/opencv.done ]]; then
    # OpenCV 4.2.0 + OpenCV contrib
    echo "Installing OpenCV 4.2.0..."

    # Determine command string to use
    if [[ "$OSTYPE" == "linux-gnu" ]]; then
        # Linux
        CMDS=""
        # ...
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        # Mac OSX
        CMDS="-DWITH_CUDA=OFF"
    fi

    cd $BUILD_DIR
    rm -rf opencv opencv_contrib ceres-solver
    # Forcing versions below CUDA 5.3 to not be built because OpenCV 4.2 DNN does not support them
    git clone -b 4.2.0 https://github.com/opencv/opencv --depth 1 && \
        git clone -b 4.2.0  https://github.com/opencv/opencv_contrib \
            --depth 1 && \
        git clone -b 1.14.0 https://github.com/ceres-solver/ceres-solver \
            --depth 1 && \
        cd ceres-solver && mkdir -p build_cmake && cd build_cmake && \
        cmake .. -DCMAKE_INSTALL_PREFIX=$INSTALL_PREFIX && \
        make install -j$cores && \
        mkdir -p $BUILD_DIR/opencv/build && cd $BUILD_DIR/opencv/build && \
        cmake -D CMAKE_BUILD_TYPE=Release \
              -D CMAKE_INSTALL_PREFIX=$INSTALL_PREFIX \
              -D BUILD_TESTS=OFF -D BUILD_PERF_TESTS=OFF -D ENABLE_FAST_MATH=1 \
              -D WITH_CUDA=ON -D CUDA_FAST_MATH=1 -D WITH_CUBLAS=1 -D WITH_NVCUVID=1 \
              -D BUILD_opencv_rgbd=OFF \
              -D BUILD_opencv_cnn_3dobj=OFF \
              -D BUILD_opencv_cudacodec=OFF \
              -D BUILD_opencv_xfeatures2d=OFF \
              -D CUDA_ARCH_BIN="5.3 6.0 6.1" \
              -D WITH_PROTOBUF=OFF \
              -D BUILD_PROTOBUF=OFF \
              -D OPENCV_EXTRA_MODULES_PATH=$BUILD_DIR/opencv_contrib/modules \
              -D WITH_INF_ENGINE=ON \
              -D ENABLE_CXX11=ON \
              -D BUILD_opencv_dnn=$BUILD_CV_DNN \
              -D OPENCV_ENABLE_NONFREE=ON \
              -D WITH_PROTOBUF=ON \
              -D BUILD_PROTOBUF=ON \
              -D BUILD_LIBPROTOBUF_FROM_SOURCES=OFF \
              $(echo $CMDS) -DCMAKE_PREFIX_PATH=$(echo $PY_EXTRA_CMDS) ..
        make install -j$cores && touch $BUILD_DIR/opencv.done \
            || { echo 'Installing OpenCV failed!' ; exit 1; }
    echo "Done installing OpenCV 4.2.0"
fi

if [[ $INSTALL_GRPC == true ]] && [[ ! -f $BUILD_DIR/grpc.done ]] ; then
    # gRPC 1.16.0
    echo "Installing gRPC 1.16.0..."
    cd $BUILD_DIR
    rm -fr grpc
    git clone -b v1.16.0 --depth 1 https://github.com/grpc/grpc && \
        cd grpc && git submodule update --init --recursive && \
        CPPFLAGS=-I$INSTALL_PREFIX/include LDFLAGS=-L$INSTALL_PREFIX/lib make -j$cores && \
        CPPFLAGS=-I$INSTALL_PREFIX/include LDFLAGS=-L$INSTALL_PREFIX/lib make install prefix=$INSTALL_PREFIX && \
        touch $BUILD_DIR/grpc.done \
            || { echo 'Installing gRPC failed!' ; exit 1; }
    if [[ "$OSTYPE" == "linux-gnu" ]]; then
        # Linux
        ldconfig -n $INSTALL_PREFIX/lib
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        # OS X
        install_name_tool -id "@rpath/libgrpc++_unsecure.dylib" \
                          $INSTALL_PREFIX/lib/libgrpc++_unsecure.dylib
        install_name_tool -id "@rpath/libgrpc.dylib" \
                          $INSTALL_PREFIX/lib/libgrpc.dylib
        install_name_tool -id "@rpath/libgpr.dylib" \
                          $INSTALL_PREFIX/lib/libgpr.dylib
        install_name_tool -change libgpr.dylib @rpath/libgpr.dylib \
                          $INSTALL_PREFIX/lib/libgrpc++_unsecure.dylib
        install_name_tool -change libgrpc_unsecure.dylib @rpath/libgrpc_unsecure.dylib \
                          $INSTALL_PREFIX/lib/libgrpc++_unsecure.dylib
    fi
    echo "Done installing gRPC 1.16.0"
fi

if [[ $INSTALL_FFMPEG == true ]] && [[ ! -f $BUILD_DIR/ffmpeg.done ]] ; then
    echo "Installing ffmpeg 4.2..."

    # Determine command string to use
    if [[ "$OSTYPE" == "linux-gnu" ]]; then
        # Linux
        CMDS="--extra-version=0ubuntu0.16.04.1
              --toolchain=hardened
              --cc=cc --cxx=g++"
        # ...
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        # Mac OSX
        CMDS=""
    fi

    # FFMPEG
    cd $BUILD_DIR
    rm -fr ffmpeg
    git clone -b n4.2 https://git.ffmpeg.org/ffmpeg.git && cd ffmpeg && \
    ./configure --prefix=$INSTALL_PREFIX \
                --enable-shared --disable-stripping \
                --disable-decoder=libschroedinger \
                --enable-avresample \
                --enable-libx264 \
                --enable-nonfree \
                --enable-gpl \
                --enable-gnutls \
		--enable-libx265 \
                $(echo $CMDS) && \
    make -j${cores} && make install && touch $BUILD_DIR/ffmpeg.done \
        || { echo 'Installing ffmpeg failed!' ; exit 1; }
    echo "Done installing ffmpeg 4.2"
fi

if [[ $INSTALL_PYBIND == true ]] && [[ ! -f $BUILD_DIR/pybind.done ]] ; then
    echo "Installing pybind..."
    cd $BUILD_DIR
    rm -fr pybind11
    git clone -b v2.2.4 https://github.com/pybind/pybind11 --depth 1 && \
        cd pybind11 && \
        mkdir build && cd build && \
        cmake .. -DCMAKE_INSTALL_PREFIX=$INSTALL_PREFIX -DPYBIND11_TEST=Off -DCMAKE_BUILD_TYPE=RelWithDebInfo && \
        make install -j${cores} && cd ../../ && \
        touch $BUILD_DIR/pybind.done \
            || { echo 'Installing pybind failed!' ; exit 1; }
    echo "Done installing pybind"
fi

if [[ $INSTALL_STOREHOUSE == true ]] && [[ ! -f $BUILD_DIR/storehouse.done ]] ; then
    echo "Installing storehouse..."
    if [ $ROOT_INSTALL == true ]; then
        BUILD_CMD="cd python && rm -rf dist && python3 setup.py bdist_wheel && cwd=\$(pwd) && pushd /tmp && ((yes | pip3 uninstall storehouse) || true) && (yes | pip3 install \$cwd/dist/*) && popd"
    else
        BUILD_CMD="./build.sh"
    fi
    cd $BUILD_DIR
    rm -fr storehouse
    git clone https://github.com/scanner-research/storehouse && \
        cd storehouse && \
        git checkout v0.6.3 && \
        cd thirdparty && mkdir build && cd build && \
        cmake .. -DCMAKE_INSTALL_PREFIX=$INSTALL_PREFIX -DCMAKE_BUILD_TYPE=RelWithDebInfo && \
        make -j${cores} && cd ../../ && \
        mkdir build && cd build && \
        cmake .. -DCMAKE_INSTALL_PREFIX=$INSTALL_PREFIX -DCMAKE_BUILD_TYPE=RelWithDebInfo && \
        make install -j${cores} && cd .. && \
        CPATH=$INSTALL_PREFIX/include LD_LIBRARY_PATH=$INSTALL_PREFIX/lib ./build.sh && \
        touch $BUILD_DIR/storehouse.done \
            || { echo 'Installing storehouse failed!' ; exit 1; }
    echo "Done installing storehouse"
fi

if [[ $INSTALL_GOOGLETEST == true ]] && [[ ! -f $BUILD_DIR/googletest.done ]]; then
    echo "Installing googletest..."
    cd $BUILD_DIR
    rm -fr googletest
    git clone https://github.com/google/googletest && \
        cd googletest && git checkout release-1.8.1 && \
        mkdir build && cd build && \
        cmake .. -DCMAKE_INSTALL_PREFIX=$INSTALL_PREFIX && \
        make -j${cores} && make install && \
        touch $BUILD_DIR/googletest.done \
            || { echo 'Installing googletest failed!' ; exit 1; }
    echo "Done installing googletest"
fi

if [[ $INSTALL_HWANG == true ]] && [[ ! -f $BUILD_DIR/hwang.done ]] ; then
    echo "Installing hwang..."
    if [ $ROOT_INSTALL == true ]; then
        BUILD_CMD="cd python && rm -rf dist && python3 setup.py bdist_wheel && cwd=\$(pwd) && pushd /tmp && ((yes | pip3 uninstall hwang) || true) && (yes | pip3 install \$cwd/dist/*) && popd"
    else
        BUILD_CMD="./build.sh"
    fi
       
    cd $BUILD_DIR
    rm -fr hwang
    git clone https://github.com/scanner-research/hwang && \
        cd hwang && \
        git checkout v0.5.1 && \
        bash ./deps.sh -a \
             --with-ffmpeg $INSTALL_PREFIX \
             --with-protobuf $INSTALL_PREFIX \
             --cores ${cores} && \
        mkdir -p build && cd build && \
        cmake .. -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_INSTALL_PREFIX=$INSTALL_PREFIX -DBUILD_CUDA=$USE_GPU && \
        make install -j${cores} && \
        cd .. && \
        eval $BUILD_CMD && \
        touch $BUILD_DIR/hwang.done \
            || { echo 'Installing hwang failed!' ; exit 1; }
    echo "Done installing hwang"
fi

if [[ $INSTALL_TINYTOML == true ]] && [[ ! -f $BUILD_DIR/tinytoml.done ]]; then
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

if [[ $INSTALL_HALIDE == true ]] && [[ ! -f $BUILD_DIR/halide.done ]] ; then
    # Halide
    echo "Installing Halide..."

    cd $BUILD_DIR
    rm -fr Halide
    mkdir Halide
    cd Halide
    if [[ "$OSTYPE" == "linux-gnu" ]]; then
        # If CLANG is not set, we should set it to clang or clang-5.0
        if [ -z ${CLANG+x} ]; then
            if command -v clang >/dev/null 2>&1 &&
                   [[ $(clang++ -v 2>&1 |
                         grep version |
                         sed 's/.*version \([0-9]*.[0-9]*.[0-9]*\) .*/\1/g' |
                         perl -pe '($_)=/([0-9]+([.][0-9]+)+)/') > '4.0.0' ]]; then
                export CLANG=clang
            elif command -v clang-5.0 >/dev/null 2>&1; then
                export CLANG=clang-5.0
            fi
            echo $CLANG
        fi
        # If LLVM_CONFIG is not set, we should set it to llvm-config or
        # llvm-config-5.0
        if [ -z ${LLVM_CONFIG+x} ]; then
            if command -v llvm-config >/dev/null 2>&1 &&
                   [[ $(llvm-config --version) > '4.0.0' ]]; then
                export LLVM_CONFIG=llvm-config
            elif command -v llvm-config-5.0 >/dev/null 2>&1; then
                export LLVM_CONFIG=llvm-config-5.0
            fi
        fi
        git clone -b release_2018_02_15 https://github.com/halide/Halide --depth 1 && \
            cd Halide && \
            make distrib -j$cores && \
            cp -r distrib/* $INSTALL_PREFIX && \
            touch $BUILD_DIR/halide.done \
                || { echo 'Installing Halide failed!' ; exit 1; }
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        TAR_NAME=halide-mac-64-trunk-46d8e9e0cdae456489f1eddfd6d829956fc3c843.tgz
        wget --retry-on-http-error=403 https://github.com/halide/Halide/releases/download/release_2018_02_15/$TAR_NAME && \
            wget --retry-on-http-error=403 https://raw.githubusercontent.com/halide/Halide/release_2018_02_15/src/Generator.h && \
            tar -zxf $TAR_NAME && \
            cp Generator.h halide/include && \
            mkdir -p $INSTALL_PREFIX/lib && \
            find ./halide -type f -exec chmod 644 {} + && \
            find ./halide -type d -exec chmod 755 {} + && \
            find ./halide/bin -type f -exec chmod 755 {} + && \
            cp -r halide/bin/* $INSTALL_PREFIX/lib && \
            rm -r halide/bin && \
            cp -r halide/* $INSTALL_PREFIX && \
            install_name_tool -id "@rpath/libHalide.dylib" $INSTALL_PREFIX/lib/libHalide.dylib
            touch $BUILD_DIR/halide.done \
                || { echo 'Installing Halide failed!' ; exit 1; }
    fi

    echo "Done installing Halide"
fi

if [[ $INSTALL_CAFFE == true ]] && [[ $USE_GPU == false ]] && \
       [[ "$OSTYPE" == "linux-gnu" ]] && [[ ! -f $BUILD_DIR/caffe.done ]]; then
    # Intel Caffe 1.0.6
    cd $BUILD_DIR
    rm -fr caffe
    # Use more recent mkldnn commit to fix gcc bug
    git clone -b 1.0.6 https://github.com/intel/caffe --depth 1 && \
        cd caffe && \
        cp $FILES_DIR/caffe/Makefile.config Makefile.config && \
        rm mkldnn.commit && \
        echo "2604f435da7bb9f1896ae37200d91734adfdba9c" > mkldnn.commit && \
        mkdir build && cd build && \
        cmake -DCMAKE_INSTALL_PREFIX=$INSTALL_PREFIX \
              -DCMAKE_PREFIX_PATH="$INSTALL_PREFIX;$PY_EXTRA_CMDS" \
              -DCPU_ONLY=ON \
              -DOpenCV_DIR=$OPENCV_DIR \
              -DUSE_OPENCV=OFF \
              -DBUILD_python=OFF \
              -Dpython_version=3 \
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

#if [[ $INSTALL_CAFFE == true ]] &&
if [[ $INSTALL_CAFFE == true ]] && \
       ([[ $USE_GPU == true ]] ||
        [[ "$OSTYPE" == "darwin"* ]]) && \
       [[ ! -f $BUILD_DIR/caffe.done ]]; then
    cd $BUILD_DIR
    # Intel MKL

    if [[ "$OSTYPE" == "linux-gnu" ]]; then
        rm -fr mkl
        mkdir mkl && \
            cd mkl && \
            wget http://registrationcenter-download.intel.com/akdlm/irc_nas/tec/16318/l_mkl_2020.0.166.tgz && \
            tar -zxf l_mkl_2020.0.166.tgz && \
            cp $FILES_DIR/mkl/silent.cfg silent.cfg && \
            echo "PSET_INSTALL_DIR=$INSTALL_PREFIX/intel" >> silent.cfg && \
            cd l_mkl_2020.0.166 && \
            bash install.sh --cli-mode --silent ../silent.cfg
    fi

    if [[ $USE_GPU == true ]]; then
        CPU_ONLY=OFF
    else
        CPU_ONLY=ON
    fi

    cd $BUILD_DIR
    # Caffe rc5
    rm -fr caffe
    git clone https://github.com/BVLC/caffe && \
        cd caffe &&
        git checkout 18b09e807a6e146750d84e89a961ba8e678830b4 &&
        cp $FILES_DIR/caffe/Makefile.config Makefile.config && \
        mkdir build && cd build && \
        cmake -DCMAKE_INSTALL_PREFIX=$INSTALL_PREFIX \
              -DCMAKE_PREFIX_PATH=$INSTALL_PREFIX \
              -DINTEL_ROOT=$INSTALL_PREFIX/intel \
              -DCPU_ONLY=$CPU_ONLY \
              -DBLAS=mkl \
              -DBUILD_python=OFF \
              -Dpython_version=3 \
              -DCUDA_ARCH_NAME="Manual" \
              -DCUDA_ARCH_BIN="30 35 50 60 61" \
              -DCUDA_ARCH_PTX="30 35 50 60 61" \
              -DOpenCV_DIR=$INSTALL_PREFIX \
              -DUSE_OPENCV=OFF \
              -DCMAKE_CXX_FLAGS="-std=c++11" \
              .. && \
        make -j${cores} && \
        make install && \
        touch $BUILD_DIR/caffe.done \
            || { echo 'Installing caffe failed!' ; exit 1; }
fi

if [[ $INSTALL_OPENPOSE == true ]] && [[ ! -f $BUILD_DIR/openpose.done ]] && \
       ! [[ "$OSTYPE" == "darwin"* ]]; then
    EXTRA_FLAGS=""
    if [[ $HAVE_GPU == false ]]; then
        EXTRA_FLAGS="-DGPU_MODE=CPU_ONLY"
    fi

    cd $BUILD_DIR
    rm -rf openpose
    git clone https://github.com/CMU-Perceptual-Computing-Lab/openpose && \
        cd openpose &&  \
        git checkout e7632893c28495f0a4af788d9c55c720be63ff2a && \
        mkdir build && cd build && \
        cmake -D CMAKE_INSTALL_PREFIX=$INSTALL_PREFIX \
              -D CMAKE_PREFIX_PATH=$INSTALL_PREFIX \
              -D OpenCV_DIR=$INSTALL_PREFIX \
              -D BUILD_CAFFE=OFF \
              -D Caffe_INCLUDE_DIRS=$CAFFE_DIR/include \
              -D Caffe_LIBS=$CAFFE_DIR/lib/libcaffe.so \
              -D BUILD_EXAMPLES=Off \
              -D BUILD_DOCS=Off \
              -D DOWNLOAD_COCO_MODEL=Off \
              -D DOWNLOAD_HAND_MODEL=Off \
              -D DOWNLOAD_FACE_MODEL=Off \
              -DCUDA_ARCH="Manual" \
              -DCUDA_ARCH_BIN="30 35 50 60 61" \
              -DCUDA_ARCH_PTX="30 35 50 60 61" \
              ${EXTRA_FLAGS} \
              .. && \
        make install -j${cores} && \
        touch $BUILD_DIR/openpose.done \
              || { echo 'Installing OpenPose failed!'; exit 1; }
fi


if [[ $INSTALL_LIBPQXX == true ]] && [[ ! -f $BUILD_DIR/libpqxx.done ]]; then
    cd $BUILD_DIR
    rm -rf libpqxx
    git clone -b 6.2.2 https://github.com/jtv/libpqxx --depth 1 && \
        cd libpqxx && \
        CXXFLAGS="-fPIC" ./configure --prefix=$INSTALL_PREFIX --disable-documentation && \
        make install -j${cores} && \
        touch $BUILD_DIR/libpqxx.done \
              || { echo 'Installing libpqxx failed!'; exit 1; }
fi


DEP_FILE=$LOCAL_DIR/dependencies.txt
rm -f $DEP_FILE
echo "HAVE_GPU=$HAVE_GPU" >> $DEP_FILE
echo "CAFFE_GPU=$USE_GPU" >> $DEP_FILE

echo "OpenCV_DIR=$OPENCV_DIR" >> $DEP_FILE
echo "PROTOBUF_DIR=$PROTOBUF_DIR" >> $DEP_FILE
echo "GRPC_DIR=$GRPC_DIR" >> $DEP_FILE
echo "TinyToml_DIR=$TINYTOML_DIR" >> $DEP_FILE
echo "STOREHOUSE_DIR=$STOREHOUSE_DIR" >> $DEP_FILE
echo "PYBIND11_DIR=$PYBIND_DIR" >> $DEP_FILE

echo "NO_FFMPEG=$NO_FFMPEG" >> $DEP_FILE
echo "FFMPEG_DIR=$FFMPEG_DIR" >> $DEP_FILE
echo "NO_OPENPOSE=$NO_OPENPOSE" >> $DEP_FILE
echo "NO_CAFFE=$NO_CAFFE" >> $DEP_FILE
echo "Caffe_DIR=$CAFFE_DIR" >> $DEP_FILE
echo "NO_HALIDE=$NO_HALIDE" >> $DEP_FILE
echo "Halide_DIR=$HALIDE_DIR" >> $DEP_FILE
echo "NO_LIBPQXX=$NO_LIBPQXX" >> $DEP_FILE
echo "LIBPQXX_DIR=$LIBPQXX_DIR" >> $DEP_FILE
echo "Hwang_DIR=$HWANG_DIR" >> $DEP_FILE
echo "NO_HWANG=$NO_HWANG" >> $DEP_FILE

echo
echo "Done installing dependencies! Add the following to your shell config:"
echo
echo "export LD_LIBRARY_PATH=$INSTALL_PREFIX/lib:\$LD_LIBRARY_PATH"
echo "export PATH=$INSTALL_PREFIX/bin:\$PATH"
echo "export PKG_CONFIG_PATH=$INSTALL_PREFIX/lib/pkgconfig:\$PKG_CONFIG_PATH"
if [[ $INSTALL_OPENCV == true ]]; then
    echo "export PYTHONPATH=$INSTALL_PREFIX/lib/python$PYTHON_VERSION/dist-packages:\$PYTHONPATH"
fi
if [[ $INSTALL_CAFFE_CPU == true ]] || [[ $INSTALL_CAFFE_GPU == true ]]; then
    echo "export PYTHONPATH=$INSTALL_PREFIX/python:\$PYTHONPATH"
fi
if [[ $INSTALL_OPENVINO == true ]]; then
    INTEL_OPENVINO_DIR=$INSTALL_PREFIX/intel/openvino_2019.3.376
    echo "export INTEL_OPENVINO_DIR=$INTEL_OPENVINO_DIR"
    echo "export INTEL_CVSDK_DIR=$INTEL_OPENVINO_DIR"
    echo "export InferenceEngine_DIR=$INTEL_OPENVINO_DIR/deployment_tools/inference_engine/share"
    echo "export IE_PLUGINS_PATH=$INTEL_OPENVINO_DIR/deployment_tools/inference_engine/lib/intel64"
    echo "export HDDL_INSTALL_DIR=$INTEL_OPENVINO_DIR/deployment_tools/inference_engine/external/hddl"
    HDDL_INSTALL_DIR=$INTEL_OPENVINO_DIR/deployment_tools/inference_engine/external/hddl
    IE_PLUGINS_PATH=$INTEL_OPENVINO_DIR/deployment_tools/inference_engine/lib/intel64
    echo "export LD_LIBRARY_PATH=$HDDL_INSTALL_DIR/lib:$INTEL_OPENVINO_DIR/deployment_tools/inference_engine/external/gna/lib:$INTEL_OPENVINO_DIR/deployment_tools/inference_engine/external/mkltiny_lnx/lib:$INTEL_OPENVINO_DIR/deployment_tools/inference_engine/external/tbb/lib:$IE_PLUGINS_PATH:\$LD_LIBRARY_PATH"
    # OpenVINO's Python API
    echo "export PYTHONPATH=$INTEL_OPENVINO_DIR/python/python$PYTHON_VERSION:\$PYTHONPATH"
fi

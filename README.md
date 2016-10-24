# Scanner

## Building

You will need [OpenCV](https://github.com/opencv/opencv) (2.4.x or 3.x, 3.x preferred) built with [opencv_contrib](https://github.com/opencv/opencv_contrib/).

### OS X Dependencies
```
brew install openssl curl webp homebrew/science/opencv3 ffmpeg mpich
```
### Ubuntu Dependencies

```
sudo apt-get install -y libopenssl-dev libcurl3-dev liblzma-dev libprotobuf-dev protobuf-compiler \
     libeigen3-dev libgflags-dev libgoogle-glog-dev libatlas-base-dev libsuitesparse-dev \
     libx264-dev libavcodec-dev libavresample-dev libavformat-dev libavfilter-dev ffmpeg \
     libbpng-dev libjpeg-dev libbz2-dev cmake
```

### Python dependencies

```
pip install numpy protobuf toml
```

### Building the project

```
git clone https://github.com/apoms/scanner && cd scanner
mkdir build && cd build
cmake -D PIPELINE_FILE=../scanner/pipelines/sample_pipeline.cpp ..
```

## Building the results server
Enable the CMake flag `-DBUILD_SERVER=ON`.

### OS X Dependencies
#### Installing Folly
https://github.com/facebook/folly

Last I checked, the Homebrew formula does not work correctly with proxygen.
#### Installing Wangle
https://github.com/facebook/wangle
#### Installing Proxygen
https://dalzhim.wordpress.com/2016/04/27/compiling-facebooks-proxygen-on-os-x/

### Ubuntu Dependencies

#### Installing Proxygen
https://github.com/facebook/proxygen
```
git clone https://github.com/facebook/proxygen
cd proxygen/proxygen
sudo ./deps.sh && sudo ./reinstall
```

## Building UI

### Ubuntu
```
sudo apt-get install npm nodejs
sudo ln -s `which nodejs` /usr/bin/node
cd www
npm install
./node_modules/webpack/bin/webpack.js
```

### OS X
```
brew install npm
cd www
npm install
./node_modules/webpack/bin/webpack.js
```

## Contributing

Before committing, please install the Git hooks for this project.

```
git clone https://github.com/willcrichton/scanner-hooks
<edit files>
./scanner-hooks/install_hooks.sh </path/to/scanner>
```

You'll need to edit `pre-commit-clang-format` and change `CLANG_FORMAT` to point your `clang-format` binary. Must be version 4.0 or above.
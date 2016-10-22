# Scanner

## Building

### OS X Dependencies
```
brew install openssl curl webp homebrew/science/opencv3 ffmpeg mpich
```
### Ubuntu Dependencies

## General dependencies

```
sudo apt-get install -y libopenssl-dev libcurl3-dev liblzma-dev libprotobuf-dev protobuf-compiler libeigen3-dev libgflags-dev libgoogle-glog-dev libatlas-base-dev libsuitesparse-dev libx264-dev libavcodec-dev libavresample-dev libavformat-dev libavfilter-dev ffmpeg libbpng-dev libjpeg-dev libbz2-dev
pip install numpy protobuf
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

# Scanner

## Building

### OS X Dependencies
```
brew install openssl curl webp homebrew/science/opencv3 ffmpeg mpich
```
### Ubuntu Dependencies

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

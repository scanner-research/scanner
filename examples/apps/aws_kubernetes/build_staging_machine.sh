# Install docker
sudo apt-get install -y \
    apt-transport-https \
    ca-certificates \
    curl \
    software-properties-common

curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
sudo add-apt-repository \
   "deb [arch=amd64] https://download.docker.com/linux/ubuntu \
   $(lsb_release -cs) \
   stable"
sudo apt-get update
sudo apt-get install -y docker-ce
sudo usermod -a -G docker $USER

# Install jq
sudo apt-get install -y jq

# Install aws
sudo apt-get install -y python-pip
pip install awscli --upgrade --user

# Configure aws
aws configure

# Install kubectl
sudo apt-get update && sudo apt-get install -y apt-transport-https
curl -s https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
sudo touch /etc/apt/sources.list.d/kubernetes.list
echo "deb http://apt.kubernetes.io/ kubernetes-xenial main" | sudo tee -a /etc/apt/sources.list.d/kubernetes.list
sudo apt-get update
sudo apt-get install -y kubectl

# Install heptio auth
curl -o heptio-authenticator-aws https://amazon-eks.s3-us-west-2.amazonaws.com/1.10.3/2018-06-05/bin/linux/amd64/heptio-authenticator-aws
chmod +x ./heptio-authenticator-aws
mkdir -p ~/bin
cp heptio-authenticator-aws ~/bin
echo 'export PATH=$HOME/bin:$PATH' >> ~/.bashrc
export PATH=$HOME/bin:$PATH

# Install scanner
sudo apt-get install -y \
   build-essential \
   cmake git libgtk2.0-dev pkg-config unzip llvm-5.0-dev clang-5.0 libc++-dev \
   libgflags-dev libgtest-dev libssl-dev libcurl3-dev liblzma-dev \
   libeigen3-dev libgoogle-glog-dev libatlas-base-dev libsuitesparse-dev \
   libgflags-dev libx264-dev libopenjpeg-dev libxvidcore-dev \
   libpng-dev libjpeg-dev libbz2-dev wget \
   libleveldb-dev libsnappy-dev libhdf5-serial-dev liblmdb-dev python-dev \
   python-tk autoconf autogen libtool libtbb-dev libopenblas-dev \
   liblapacke-dev swig yasm python3.5 python3-pip cpio automake libass-dev \
   libfreetype6-dev libsdl2-dev libtheora-dev libtool \
   libva-dev libvdpau-dev libvorbis-dev libxcb1-dev libxcb-shm0-dev \
   libxcb-xfixes0-dev mercurial texinfo zlib1g-dev curl libcap-dev \
   libboost-all-dev libgnutls-dev libpq-dev postgresql libx265-dev

pip3 install numpy

git clone https://github.com/scanner-research/scanner.git
cd scanner
sudo bash ./deps.sh -a --prefix /usr/local
mkdir build
cd build
cmake ..
make -j$(nproc)
cd ..
bash ./build.sh

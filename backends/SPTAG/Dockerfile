FROM mcr.microsoft.com/oss/mirror/docker.io/library/ubuntu:20.04
WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get -y install wget build-essential swig cmake git libnuma-dev python3.8-dev python3-distutils gcc-8 g++-8 \
    libboost-filesystem-dev libboost-test-dev libboost-serialization-dev libboost-regex-dev libboost-serialization-dev libboost-regex-dev libboost-thread-dev libboost-system-dev libtbb-dev

RUN wget https://bootstrap.pypa.io/pip/3.8/get-pip.py && python3.8 get-pip.py && python3.8 -m pip install numpy

ENV PYTHONPATH=/app/Release
ENV LD_LIBRARY_PATH=/app/Release:${LD_LIBRARY_PATH}

COPY CMakeLists.txt ./
COPY AnnService ./AnnService/
COPY Test ./Test/
COPY Wrappers ./Wrappers/
COPY GPUSupport ./GPUSupport/
COPY ThirdParty ./ThirdParty/

RUN rm -rf ThirdParty/zstd && mkdir -p ThirdParty/zstd && cd ThirdParty/zstd && git clone -b release https://github.com/facebook/zstd.git . && \
    export CC=/usr/bin/gcc-8 && export CXX=/usr/bin/g++-8 && cd /app && mkdir -p build && cd build && cmake -DSPDK=OFF -DROCKSDB=OFF .. && make -j && cd ..

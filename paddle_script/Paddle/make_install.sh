#!/bin/bash

# make
echo "make"
cd build
cmake .. -DPY_VERSION=3.7 -DWITH_GPU=ON -DWITH_DISTRIBUTE=OFF -DWITH_TESTING=ON -DWITH_INFERENCE_API_TEST=OFF -DON_INFER=OFF -DCMAKE_BUILD_TYPE=Release
make -j20

if [ $? -ne 0 ];then
    echo "Make Failed"
    exit
fi

# install 
echo "install"
cd python/dist
pip3.7 uninstall -y paddlepaddle-gpu
pip3.7 install -U paddlepaddle_gpu-0.0.0-cp37-cp37m-linux_x86_64.whl
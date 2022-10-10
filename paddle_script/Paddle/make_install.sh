#!/bin/bash -ex
echo $(date '+%Y-%m-%d %H:%M:%S')

docker_id=$(cat /proc/self/cgroup | grep /docker | head -1 | cut -d/ -f3)

build_dir=${1:-build}

if [ $build_dir == "build" -a ${docker_id} != 'e0234af559f641523d2c28673242945b379e90363b5ae919d42ee72043e7e05e' ];then
    echo "Please Using Docker jc-dev-112 for compile!"
    exit
fi

if [[ ! -n `command -v sccache` ]]; then
    echo "sccache not installed, installing"
    bash /Paddle/sccache_install.sh
    if [[ $? -ne 0 ]]; then
        echo "sccache installing failed, please check or remove sccache flag"
    fi
fi

# make
echo "make"
mkdir -p ${build_dir}
cd ${build_dir}

cmake .. \
    -DWITH_PYTHON=ON \
    -DPY_VERSION=3.7 \
    -DWITH_GPU=ON \
    -DCUDA_ARCH_NAME=Auto \
    -DWITH_DISTRIBUTE=ON \
    -DWITH_TESTING=ON \
    -DWITH_PROFILER=ON \
    -DWITH_INFERENCE_API_TEST=ON \
    -DON_INFER=ON \
    -DCMAKE_BUILD_TYPE=Release \
    -DWITH_MKL=ON \
    -DCMAKE_CXX_COMPILER_LAUNCHER=sccache \
    -DCMAKE_CXX_LINK_LANCHER=sccache \
    -DWITH_CINN=OFF \
    -DCINN_GIT_TAG=develop

if [ $? -ne 0 ];then
    echo "Cmake Failed"
    exit
fi

make -j30

if [ $? -ne 0 ];then
    echo "Make Failed"
    exit
fi

# install
echo "install"
cd python/dist
python -m pip uninstall -y paddlepaddle-gpu
python -m pip --no-cache-dir install -U paddlepaddle_gpu-0.0.0-cp37-cp37m-linux_x86_64.whl

echo $(date '+%Y-%m-%d %H:%M:%S')
end_time=$(TZ=Asia/Shanghai date '+%Y-%m-%d %H:%M:%S')
echo "start compile at ${start_time}"
echo "end compile at ${end_time}"

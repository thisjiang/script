#!/bin/bash -ex
echo $(date '+%Y-%m-%d %H:%M:%S')

docker_id=$(cat /proc/self/cgroup | grep /docker | head -1 | cut -d/ -f3)
if [ ${docker_id} != 'e0234af559f641523d2c28673242945b379e90363b5ae919d42ee72043e7e05e' ];then
    echo "Please Using Docker jc-dev-112 for compile!"
    exit
fi

build_dir=/Paddle/Paddle/build/

cd ${build_dir}/third_party/CINN/src/external_cinn-build
make cinnapi -j10

rm -f /usr/local/lib/python3.7/dist-packages/paddle/libs/libcinnapi.so
cp -f ${build_dir}/third_party/CINN/src/external_cinn-build/libcinnapi.so /usr/local/lib/python3.7/dist-packages/paddle/libs/libcinnapi.so

export LD_LIBRARY_PATH=${build_dir}/third_party/CINN/src/external_cinn-build/:${LD_LIBRARY_PATH}

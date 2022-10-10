#!/bin/bash -ex
start_time=$(TZ=Asia/Shanghai date '+%Y-%m-%d %H:%M:%S')
echo "start compile at ${start_time}"

build_dir="build"

mkdir -p ${build_dir}
cd ${build_dir}

cmake .. -Wno-dev

if [ $? -ne 0 ];then
    echo "Cmake Failed"
    exit
fi

make

if [ $? -ne 0 ];then
    echo "Make Failed"
    exit
fi

end_time=$(TZ=Asia/Shanghai date '+%Y-%m-%d %H:%M:%S')
echo "start compile at ${start_time}, end compile at ${end_time}"

#!/bin/bash -ex
# Copyright (c) 2022 thisjiang Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License

start_time=$(TZ=Asia/Shanghai date '+%Y-%m-%d %H:%M:%S')
echo "start compile at ${start_time}"

build_dir="build"

mkdir -p ${build_dir}
cd ${build_dir}

cmake .. \
    -Wno-dev \
    -DWITH_TEST=ON \
    -DWITH_CUDA=OFF

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

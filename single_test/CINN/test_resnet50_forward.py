# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function

import logging
import numpy as np
import paddle
import unittest

paddle.enable_static()

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


def set_cinn_flag(val):
    cinn_compiled = False
    try:
        paddle.set_flags({'FLAGS_use_cinn': val})
        cinn_compiled = True
    except ValueError:
        logger.warning("The used paddle is not compiled with CINN.")
    return cinn_compiled


@unittest.skipIf(not set_cinn_flag(True), "Paddle is not compiled with CINN.")
class TestResnet50Accuracy(unittest.TestCase):
    def reader(self, limit):
        for _ in range(limit):
            yield {'image' : np.random.randint(0, 256, size=[8, 3, 224, 224]).astype('float32'), \
                   'label' : np.random.randint(0, 1000, size=[8]).astype('int64')}

    def generate_random_data(self, loop_num=10):
        feed = []
        data = self.reader(loop_num)
        for _ in range(loop_num):
            feed.append(next(data))
        return feed

    def build_program(self, main_program, startup_program):
        with paddle.static.program_guard(main_program, startup_program):
            image = paddle.static.data(
                name='image', shape=[None, 3, 224, 224], dtype='float32')
            label = paddle.static.data(name='label', shape=[None], dtype='int64')

            model = paddle.vision.models.resnet50()
            prediction = model(image)

        return prediction

    def train(self, place, iters, feed, use_cinn=False, seed=1234):
        np.random.seed(seed)
        paddle.seed(seed)
        if paddle.is_compiled_with_cuda():
            # paddle.set_flags({'FLAGS_cudnn_deterministic': 1})
            paddle.fluid.set_flags({'FLAGS_cudnn_exhaustive_search': 1})
        set_cinn_flag(use_cinn)

        startup_program = paddle.static.Program()
        main_program = paddle.static.Program()

        prediction = self.build_program(main_program, startup_program)
        exe = paddle.static.Executor(place)

        build_strategy = paddle.static.BuildStrategy()
        if use_cinn == True:
            build_strategy.debug_graphviz_path = "./viz_file/"

        parallel_exec = paddle.static.CompiledProgram(
            main_program, build_strategy).with_data_parallel()
        pred_vals = []
        scope = paddle.static.Scope()

        with paddle.static.scope_guard(scope):
            exe.run(startup_program)
            for step in range(iters):
                pred_v = exe.run(parallel_exec,
                                 feed=feed[step],
                                 fetch_list=[prediction],
                                 return_numpy=True)
                pred_vals.append(pred_v[0][0])
        return pred_vals

    def test_check_resnet50_accuracy(self):
        place = paddle.CUDAPlace(0) if paddle.is_compiled_with_cuda(
        ) else paddle.CPUPlace()

        loop_num = 10
        feed = self.generate_random_data(loop_num)

        pred_c = self.train(place, loop_num, feed, use_cinn=True)
        pred_p = self.train(place, loop_num, feed, use_cinn=False)

        print("Open CINN prediction value: ", pred_c)
        print("Close CINN  prediction value: ", pred_p)

        self.assertTrue(np.allclose(pred_c, pred_p, atol=1e-5))


if __name__ == '__main__':
    unittest.main()

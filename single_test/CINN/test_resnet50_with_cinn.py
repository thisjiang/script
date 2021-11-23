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
import sys

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
            yield {'image': np.random.randint(0, 256, size=[8, 3, 224, 224]).astype('float32'), \
                   'label': np.random.randint(0, 1000, size=[8]).astype('int64')}

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

            image.stop_gradient = False

            model = paddle.vision.models.resnet50()
            prediction = model(image)

            loss = paddle.nn.functional.cross_entropy(
                input=prediction, label=label)
            loss = paddle.mean(loss)
            adam = paddle.optimizer.Adam(learning_rate=0.001)
            adam.minimize(loss)
        return loss, prediction

    def train(self, place, iters, feed, use_cinn=False, seed=1234):
        np.random.seed(seed)
        paddle.seed(seed)
        if paddle.is_compiled_with_cuda():
            paddle.set_flags({'FLAGS_cudnn_deterministic': 1})
        set_cinn_flag(use_cinn)
        paddle.set_flags({'FLAGS_paddle_num_threads': 1})

        startup_program = paddle.static.Program()
        main_program = paddle.static.Program()

        loss, prediction = self.build_program(main_program, startup_program)
        exe = paddle.static.Executor(place)

        build_strategy = paddle.static.BuildStrategy()
        build_strategy.remove_unnecessary_lock = False
        exec_strategy = paddle.static.ExecutionStrategy()
        exec_strategy.num_threads = 1
        exec_strategy.num_iteration_per_drop_scope = 100

        compiled_prog = paddle.static.CompiledProgram(
            main_program).with_data_parallel(loss_name=loss.name,
            build_strategy = build_strategy,
            exec_strategy = exec_strategy)

        loss_vals = []
        pred_vals = []
        scope = paddle.static.Scope()

        with paddle.static.scope_guard(scope):
            exe.run(startup_program)
            print("Running step ", file=sys.stderr, end=' ')
            for step in range(iters):
                print(step, file=sys.stderr, end=' ')
                loss_v, pred_v = exe.run(compiled_prog,
                                 feed=feed[step],
                                 fetch_list=[loss, prediction],
                                 return_numpy=True)
                loss_vals.append(loss_v[0])
                pred_vals.append(pred_v[0][0])
        return loss_vals, pred_vals

    def test_check_resnet50_accuracy(self):
        place = paddle.CUDAPlace(0) if paddle.is_compiled_with_cuda(
        ) else paddle.CPUPlace()

        np.set_printoptions(suppress=True)

        loop_num = 100
        feed = self.generate_random_data(loop_num)

        loss_c, pred_c = self.train(place, loop_num, feed, use_cinn=True)
        loss_p, pred_p = self.train(place, loop_num, feed, use_cinn=False)

        # print("Open CINN prediction value: ", pred_c)
        # print("Close CINN  prediction value: ", pred_p)

        # print("Open CINN loss error: ", loss_c)
        # print("Close CINN loss error: ", loss_p)

        # print("Open CINN prediction diff: ", np.fabs(np.asarray(pred_c) - np.asarray(pred_p)))
        # print("Open CINN loss diff: ", np.fabs(np.asarray(loss_c) - np.asarray(loss_p)))

        loss_c_np = np.asarray(loss_c).flatten()
        loss_p_np = np.asarray(loss_p).flatten()

        err_num = 0
        for i in range(len(loss_c_np)):
            if not np.isclose(loss_c_np[i], loss_p_np[i], atol=1e-5):
                err = np.fabs(loss_c_np[i] - loss_p_np[i])
                print("Loss value [{}, {}] diff {} at {}".format(
                        loss_c_np[i], loss_p_np[i], err, i))
                err_num += 1

        print("Total ", err_num, " loss diff greater than 1e-5")

        self.assertTrue(np.allclose(loss_c, loss_p, atol=1e-5))

        print("\n\n Loss Check OK \n\n")

        # self.assertTrue(np.allclose(pred_c, pred_p, atol=1e-5))


if __name__ == '__main__':
    unittest.main()

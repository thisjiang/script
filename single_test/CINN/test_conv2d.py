import numpy as np
import six, sys
import paddle
import logging
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
class TestConv2dAccuracy(unittest.TestCase):
  def reader(self, limit):
    for i in range(limit):
      yield { 'x' : np.random.random(size=self.shape['x']).astype('float32'), \
            'weight' : np.random.random(size=self.shape['weight']).astype('float32'), \
            'out' : np.random.randint(0, 3, size=self.shape['out']).astype('int64')}

  def RandFeedData(self, loop_num=10):
    feed = []
    data = self.reader(loop_num)
    for i in range(loop_num):
      feed.append(next(data))
    return feed

  def BuildProgram(self, main_program, startup_program):
    with paddle.static.program_guard(main_program, startup_program):
      x = paddle.static.data(name='x', shape=self.shape['x'], dtype='float32')
      weight = paddle.static.data(name='weight', shape=self.shape['weight'], dtype='float32')
      out = paddle.static.data(name='out', shape=self.shape['out'], dtype='int64')

      prediction = paddle.nn.functional.conv2d(x, weight, stride=self.shape['strides'], padding=self.shape['paddings'])

      loss = paddle.nn.functional.cross_entropy(input=prediction, label=out)
      loss = paddle.mean(loss)
      adam = paddle.optimizer.Adam()
      adam.minimize(loss)
    return loss, prediction, weight

  def Run(self, feed, use_cinn=False):
    if paddle.is_compiled_with_cuda():
        paddle.set_flags({'FLAGS_cudnn_deterministic': 1})
    set_cinn_flag(use_cinn)
    paddle.set_flags({'FLAGS_paddle_num_threads': 1})

    startup_program = paddle.static.Program()
    main_program = paddle.static.Program()

    loss, prediction, weight = self.BuildProgram(main_program, startup_program)

    exe = paddle.static.Executor(self.place)
    exe.run(startup_program)

    build_strategy = paddle.static.BuildStrategy()
    # if use_cinn == True:
    #   build_strategy.debug_graphviz_path = "./viz_file/"

    exec_strategy = paddle.static.ExecutionStrategy()
    exec_strategy.num_threads = 1

    compiled_prog = paddle.static.CompiledProgram(
            main_program).with_data_parallel(loss_name=loss.name,
            build_strategy = build_strategy,
            exec_strategy = exec_strategy)

    loss_vals = []
    pred_vals = []
    weight_vals = []
    for step in range(self.loop_num):
      loss_v, pred_v, w_v = exe.run(compiled_prog,
                          feed=feed[step],
                          fetch_list=[loss, prediction, weight],
                          return_merged=False,
                          return_numpy=True)
      loss_vals.append(loss_v[0][0])
      pred_vals.append(pred_v[0][0][0])
      weight_vals.append(w_v[0])
    return loss_vals, pred_vals, weight_vals

  def run_check_accuracy(self):
    feed = self.RandFeedData(self.loop_num)

    loss_f, pred_f, w_f = self.Run(feed, use_cinn=False)
    loss_t, pred_t, w_t = self.Run(feed, use_cinn=True)

    # '''
    # print("Paddle prediction value: ", pred_f)
    # print("CINN prediction value: ", pred_t)
    # print("Paddle Loss value: ", loss_f)
    # print("CINN Loss value: ", loss_t)

    for i in range(self.loop_num):
      w_f_np = np.asarray(w_f[i]).flatten()
      w_t_np = np.asarray(w_t[i]).flatten()
      print("Paddle weight error", np.max(np.fabs(np.asarray(feed[i]['weight']).flatten() - w_f_np)))
      print("CINN weight error", np.max(np.fabs(np.asarray(feed[i]['weight']).flatten() - w_t_np)))

    pred_f = np.asarray(pred_f).flatten()
    pred_t = np.asarray(pred_t).flatten()

    pred_diff = np.max(np.fabs(pred_f - pred_t))
    loss_diff = np.max(np.fabs(np.asarray(loss_f) - np.asarray(loss_t)))
    # print("Paddle prediction error", pred_diff)
    # print("Paddle CINN error", loss_diff)

    if pred_diff != 0.0 or loss_diff != 0.0:
      return False, pred_diff, loss_diff

    return True, pred_diff, loss_diff
    # '''

  def run_conv2d(self, batch_size = 32):
    self.inputs_ = [[batch_size, 3, 224, 224],
                    [batch_size, 64, 56, 56],
                    [batch_size, 64, 56, 56],
                    [batch_size, 64, 56, 56],
                    [batch_size, 256, 56, 56],
                    [batch_size, 256, 56, 56],
                    [batch_size, 128, 56, 56],
                    [batch_size, 128, 28, 28],
                    [batch_size, 256, 56, 56],
                    [batch_size, 512, 28, 28],
                    [batch_size, 128, 28, 28],
                    [batch_size, 512, 28, 28],
                    [batch_size, 256, 28, 28],
                    [batch_size, 256, 14, 14],
                    [batch_size, 512, 28, 28],
                    [batch_size, 1024, 14, 14],
                    [batch_size, 256, 14, 14],
                    [batch_size, 1024, 14, 14],
                    [batch_size, 512, 14, 14],
                    [batch_size, 512, 7, 7],
                    [batch_size, 1024, 14, 14],
                    [batch_size, 2048, 7, 7],
                    [batch_size, 512, 7, 7]]

    self.filters = [[64, 3, 7, 7],
                [64, 64, 1, 1],
                [64, 64, 3, 3],
                [256, 64, 1, 1],
                [64, 256, 1, 1],
                [128, 256, 1, 1],
                [128, 128, 3, 3],
                [512, 128, 1, 1],
                [512, 256, 1, 1],
                [128, 512, 1, 1],
                [128, 128, 3, 3],
                [256, 512, 1, 1],
                [256, 256, 3, 3],
                [1024, 256, 1, 1],
                [1024, 512, 1, 1],
                [256, 1024, 1, 1],
                [256, 256, 3, 3],
                [512, 1024, 1, 1],
                [512, 512, 3, 3],
                [2048, 512, 1, 1],
                [2048, 1024, 1, 1],
                [512, 2048, 1, 1],
                [512, 512, 3, 3]]

    self.strides = [[2, 2],
                    [1, 1],
                    [1, 1],
                    [1, 1],
                    [1, 1],
                    [1, 1],
                    [2, 2],
                    [1, 1],
                    [2, 2],
                    [1, 1],
                    [1, 1],
                    [1, 1],
                    [2, 2],
                    [1, 1],
                    [2, 2],
                    [1, 1],
                    [1, 1],
                    [1, 1],
                    [2, 2],
                    [1, 1],
                    [2, 2],
                    [1, 1],
                    [1, 1]]

    self.paddings = [[3, 3],
                    [0, 0],
                    [1, 1],
                    [0, 0],
                    [0, 0],
                    [0, 0],
                    [1, 1],
                    [0, 0],
                    [0, 0],
                    [0, 0],
                    [1, 1],
                    [0, 0],
                    [1, 1],
                    [0, 0],
                    [0, 0],
                    [0, 0],
                    [1, 1],
                    [0, 0],
                    [1, 1],
                    [0, 0],
                    [0, 0],
                    [0, 0],
                    [1, 1]]

    self.inputs_ = [[1, 128, 56, 56]]
    self.filters = [[128, 128, 3, 3]]
    self.strides = [[2, 2]]
    self.paddings = [[1, 1]]

    for i in range(len(self.inputs_)):
      self.shape['x'] = self.inputs_[i]
      self.shape['weight'] = self.filters[i]
      self.shape['strides'] = self.strides[i]
      self.shape['paddings'] = self.paddings[i]
      self.shape['out'] =  [self.shape['x'][0], self.shape['weight'][0], 
                            int((self.shape['x'][2] + 2*self.shape['paddings'][0] - self.shape['weight'][2])/self.shape['strides'][0]) + 1, 1]
                            # int((self.shape['x'][3] + 2*self.shape['paddings'][1] - self.shape['weight'][3])/self.shape['strides'][1]) + 1]

      status, pred_diff, loss_diff  = self.run_check_accuracy()
      if status == False:
        print("Batch size: ", batch_size)
        print("Loop num: ", i)
        print("Input shape: ", self.shape['x'])
        print("Weight shape: ", self.shape['weight'])
        print("Stride shape: ", self.shape['strides'])
        print("Padding shape: ", self.shape['paddings'])
        print("prediction error", pred_diff)
        print("loss error", loss_diff)
        print("\n", flush=True)

      sys.stdout.flush()


  def test_conv2d(self):
    self.place = paddle.CUDAPlace(0) if paddle.is_compiled_with_cuda() else paddle.CPUPlace()
    self.loop_num = 3

    self.shape = {}

    # batch_sizes = [1, 4, 8, 16, 32, 64, 128, 256]
    batch_sizes = [1]
    for bs in batch_sizes:
      self.run_conv2d(bs)

if __name__ == "__main__":
  unittest.main()

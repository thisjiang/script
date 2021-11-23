import numpy as np
import six, sys
import paddle
import paddle.fluid as fluid

paddle.enable_static()

def reader(limit):
    for i in range(limit):
        yield {'x' : np.random.random([1, 28]).astype('float32'), \
               'y' : np.random.random([1, 28]).astype('float32')}

def RandFeedData(loop_num=10):
  paddle.seed(1)
  paddle.framework.random._manual_program_seed(1)

  feed = []
  data = reader(loop_num)
  for i in range(loop_num):
    feed.append(next(data))
  return feed

def BuildProgram(main_program, startup_program):
  with paddle.static.program_guard(main_program, startup_program):
      x = paddle.static.data(name='x', shape=[1, 28], dtype='float32')
      y = paddle.static.data(name="y", shape=[1, 28], dtype='float32')

      hidden = paddle.fluid.layers.elementwise_add(x, y)
      prediction = paddle.fluid.layers.reduce_sum(hidden)

  return prediction


def Run(place, iters, feed, use_cinn=False):
  paddle.set_flags({'FLAGS_use_cinn': use_cinn})

  startup_program = paddle.static.Program()
  main_program = paddle.static.Program()

  prediction = BuildProgram(main_program, startup_program)
  exe = paddle.static.Executor(place)

  build_strategy = paddle.static.BuildStrategy()
  build_strategy.debug_graphviz_path = "./viz_file/"
  parallel_exec = paddle.static.CompiledProgram(
      main_program, build_strategy).with_data_parallel()
  pred_vals = []
  scope = paddle.static.Scope()

  with paddle.static.scope_guard(scope):
    exe.run(startup_program)
    for step in range(iters):
        print("Step %d processing" % (step))
        pred_v = exe.run(parallel_exec,
                            feed=feed[step],
                            fetch_list=[prediction],
                            return_numpy=True)
        pred_vals.append(pred_v[0][0])
  return pred_vals


if __name__ == "__main__":
  place = paddle.CPUPlace()
  # place = paddle.CUDAPlace(0)

  loop_num = 10
  feed = RandFeedData(loop_num)

  preds = Run(place, loop_num, feed, use_cinn=True)
  print(preds)

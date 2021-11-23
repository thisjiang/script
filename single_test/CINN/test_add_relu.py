import numpy as np
import six, sys
import paddle
import paddle.fluid as fluid

paddle.enable_static()

def reader(limit):
    for i in range(limit):
        yield np.random.random([1, 28]).astype('float32'), \
              np.random.random([1, 28]).astype('float32'), \
              np.random.randint(0, 2, size=[1]).astype('int64')

def RandFeedData(loop_num=10):
  paddle.seed(1)
  paddle.framework.random._manual_program_seed(1)

  feed = []
  data = reader(loop_num)
  for i in range(loop_num):
    x, y, z = next(data)
    feed.append({'x' : x, 'y' : y, 'z' : z})
  return feed

def BuildProgram(main_program, startup_program):
  with paddle.static.program_guard(main_program, startup_program):
      x = paddle.static.data(name='x', shape=[1, 28], dtype='float32')
      y = paddle.static.data(name="y", shape=[1, 28], dtype='float32')
      z = paddle.static.data(name="z", shape=[1], dtype='int64')

      hidden = paddle.add(x, y)
      prediction = paddle.nn.functional.relu(hidden)

      loss = paddle.nn.functional.cross_entropy(input=prediction, label=z)
      loss = paddle.mean(loss)
      sgd = paddle.optimizer.SGD(learning_rate=0.001)
      sgd.minimize(loss)
  return loss


def Run(place, iters, feed, use_cinn=False):
  paddle.set_flags({'FLAGS_use_cinn': use_cinn})

  startup_program = paddle.static.Program()
  main_program = paddle.static.Program()

  loss = BuildProgram(main_program, startup_program)
  exe = paddle.static.Executor(place)

  build_strategy = paddle.static.BuildStrategy()
  build_strategy.debug_graphviz_path = "./viz_file/"
  parallel_exec = paddle.static.CompiledProgram(
      main_program, build_strategy).with_data_parallel(
      loss_name=loss.name)
  loss_vals = []
  scope = paddle.static.Scope()

  with paddle.static.scope_guard(scope):
    exe.run(startup_program)
    for step in range(iters):
        print("Step %d processing" % (step))
        loss_v = exe.run(parallel_exec,
                            feed=feed[step],
                            fetch_list=[loss],
                            return_numpy=True)
        loss_vals.append(loss_v[0][0])
  return loss_vals


if __name__ == "__main__":
  # place = paddle.CPUPlace()
  place = paddle.CUDAPlace(0)

  loop_num = 1
  feed = RandFeedData(loop_num)

  loss_f = Run(place, loop_num, feed, use_cinn=False)
  print(loss_f)
  loss_t = Run(place, loop_num, feed, use_cinn=True)
  print(loss_t)

  print(np.max(np.fabs(np.asarray(loss_f) - np.asarray(loss_t))))

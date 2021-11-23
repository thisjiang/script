import numpy as np
import six, sys
import paddle
import paddle.fluid as fluid

paddle.enable_static()

def reader(limit):
    for i in range(limit):
        yield [np.random.random([2, 1, 2, 3]).astype('float32'), \
              np.random.random([2, 1, 2, 3]).astype('float32'), \
              np.random.random([2, 1, 2, 3]).astype('float32'), \
              np.random.random([2, 1, 2, 3]).astype('float32'), \
              np.random.random([2, 1, 2, 3]).astype('float32')], \
              np.random.randint(0, 3, size=[2, 1, 2]).astype('int64')

def RandFeedData(loop_num=10):
  paddle.seed(1)
  paddle.framework.random._manual_program_seed(1)

  feed = []
  data = reader(loop_num)
  for i in range(loop_num):
    x, out = next(data)
    feed.append({
      'x' : x,
      'out' : out})
  return feed

def BuildProgram(main_program, startup_program):
  with paddle.static.program_guard(main_program, startup_program):
      x = paddle.static.data(name='x', shape=[5, 2, 1, 2, 3], dtype='float32')
      x.stop_gradient = False

      out = paddle.static.data(name='out', shape=[2, 1, 2], dtype='int64')

      prediction = paddle.sum(x, training=True)

      loss = paddle.nn.functional.cross_entropy(input=prediction, label=out)
      loss = paddle.mean(loss)
      sgd = paddle.optimizer.SGD(learning_rate=0.001)
      sgd.minimize(loss)
  return loss


def Run(place, iters, feed, use_cinn=False, debug=False):
  paddle.set_flags({'FLAGS_use_cinn': use_cinn})

  startup_program = paddle.static.Program()
  main_program = paddle.static.Program()

  loss = BuildProgram(main_program, startup_program)
  exe = paddle.static.Executor(place)

  build_strategy = paddle.static.BuildStrategy()

  if debug:
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
  place = paddle.CPUPlace()
  # place = paddle.CUDAPlace(0)

  loop_num = 10
  feed = RandFeedData(loop_num)

  loss_f = Run(place, loop_num, feed, use_cinn=False)
  print(loss_f)
  loss_t = Run(place, loop_num, feed, use_cinn=True, debug=False)
  print(loss_t)

  print(np.max(np.fabs(np.asarray(loss_f) - np.asarray(loss_t))))

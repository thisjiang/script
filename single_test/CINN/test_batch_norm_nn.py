import numpy as np
import six, sys
import paddle
import paddle.fluid as fluid

paddle.enable_static()

def reader(limit):
    for i in range(limit):
        yield {'x' : np.random.random([8, 64, 112, 112]).astype('float32'), \
              'mean' : np.random.random(size=64).astype('float32'), \
              'variance' : np.random.random(size=64).astype('float32'), \
              'weight' : np.random.random(size=64).astype('float32'), \
              'bias' : np.random.random(size=64).astype('float32'), \
              'out' : np.random.randint(0, 3, size=[8, 64, 112]).astype('int64')}

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
      x = paddle.static.data(name='x', shape=[None, 64, 112, 112], dtype='float32')
      x.stop_gradient = False

      mean = paddle.static.data(name='mean', shape=[None], dtype='float32')

      variance = paddle.static.data(name='variance', shape=[None], dtype='float32')

      weight = paddle.static.data(name='weight', shape=[None], dtype='float32')
      weight.stop_gradient = False

      bias = paddle.static.data(name='bias', shape=[None], dtype='float32')
      bias.stop_gradient = False

      out = paddle.static.data(name='out', shape=[None, 64, 112], dtype='int64')

      prediction = paddle.nn.functional.batch_norm(x, mean, variance, weight, bias, training=True)

      loss = paddle.nn.functional.cross_entropy(input=prediction, label=out)
      loss = paddle.mean(loss)
      sgd = paddle.optimizer.SGD(learning_rate=0.001)
      sgd.minimize(loss)
  return loss, prediction


def Run(place, iters, feed, use_cinn=False, debug=False):
  paddle.set_flags({'FLAGS_use_cinn': use_cinn})

  startup_program = paddle.static.Program()
  main_program = paddle.static.Program()

  loss, prediction = BuildProgram(main_program, startup_program)
  exe = paddle.static.Executor(place)

  build_strategy = paddle.static.BuildStrategy()

  if debug:
    build_strategy.debug_graphviz_path = "./viz_file/"

  parallel_exec = paddle.static.CompiledProgram(
      main_program, build_strategy).with_data_parallel(
      loss_name=loss.name)
  loss_vals = []
  res_vals = []
  scope = paddle.static.Scope()

  with paddle.static.scope_guard(scope):
    exe.run(startup_program)
    for step in range(iters):
        print("Step %d processing" % (step))
        loss_v, res = exe.run(parallel_exec,
                            feed=feed[step],
                            fetch_list=[loss, prediction],
                            return_numpy=True)
        loss_vals.append(loss_v[0])
        res_vals.append(res[0][0])
  return loss_vals, res_vals


if __name__ == "__main__":
  # place = paddle.CPUPlace()
  place = paddle.CUDAPlace(0)

  loop_num = 10
  feed = RandFeedData(loop_num)

  loss_f, res_f = Run(place, loop_num, feed, use_cinn=False)
  loss_t, res_t = Run(place, loop_num, feed, use_cinn=True, debug=True)
  # '''
  print(loss_f)
  print(loss_t)
  print("loss max error:", np.max(np.fabs(np.asarray(loss_f) - np.asarray(loss_t))))
  print("forward max error", np.max(np.fabs(np.asarray(res_f) - np.asarray(res_t))))
  # '''

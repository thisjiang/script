import numpy as np
import six, sys
import paddle
import paddle.fluid as fluid

paddle.enable_static()

def reader(limit):
    for i in range(limit):
        yield {'x' : np.random.random([8, 64, 112, 112]).astype('float32'), \
              'y' : np.random.random([8, 64, 112, 112]).astype('float32'), \
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
      y = paddle.static.data(name="y", shape=[None, 64, 112, 112], dtype='float32')
      y.stop_gradient = False
      out = paddle.static.data(name='out', shape=[None, 64, 112], dtype='int64')

      param_attr = paddle.ParamAttr(initializer=paddle.nn.initializer.Constant(value=0.5))
      bn1 = paddle.static.nn.batch_norm(x, param_attr=param_attr, bias_attr=param_attr)
      bn2 = paddle.static.nn.batch_norm(y, param_attr=param_attr, bias_attr=param_attr)

      hidden = paddle.add(bn1, bn2)
      prediction = paddle.nn.functional.relu(hidden)

      loss = paddle.nn.functional.cross_entropy(input=prediction, label=out)
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

  loop_num = 10
  feed = RandFeedData(loop_num)

  loss_t = Run(place, loop_num, feed, use_cinn=True)
  loss_f = Run(place, loop_num, feed, use_cinn=False)

  print("Close CINN: ", loss_f)
  print("Open CINN: ", loss_t)
  print("Accuracy: ", np.max(np.fabs(np.asarray(loss_f) - np.asarray(loss_t))))

import numpy as np
import six, sys
import paddle

paddle.enable_static()

def reader(limit):
    for i in range(limit):
        yield np.random.random(size=(128, 16)).astype('float32')

def RandFeedData(loop_num=10):
  feed = []
  data = reader(loop_num)
  for i in range(loop_num):
    x = next(data)
    feed.append({'x' : x})
  return feed

def BuildProgram(main_program, startup_program):
  with paddle.static.program_guard(main_program, startup_program):
    x = paddle.static.data(name='x', shape=[None, 16], dtype='float32')

    param_attr = paddle.ParamAttr(initializer=paddle.nn.initializer.Constant(value=0.5))
    prediction = paddle.static.nn.fc(x, 16, weight_attr=param_attr, bias_attr=param_attr)

    loss = paddle.mean(prediction)
    adam = paddle.optimizer.Adam()
    adam.minimize(loss)
  return loss

def Run(place, iters, feed, use_cinn=False):
  paddle.set_flags({'FLAGS_use_cinn': use_cinn})

  startup_program = paddle.static.Program()
  main_program = paddle.static.Program()

  loss = BuildProgram(main_program, startup_program)

  exe = paddle.static.Executor(place)
  exe.run(startup_program)

  build_strategy = paddle.static.BuildStrategy()
  build_strategy.debug_graphviz_path = "./viz_file/"
  parallel_exec = paddle.static.CompiledProgram(
      main_program, build_strategy).with_data_parallel(
      loss_name=loss.name)

  loss_vals = []
  for step in range(iters):
      print("Step %d processing" % (step))

      loss_v = exe.run(parallel_exec,
                          feed=feed[step],
                          fetch_list=[loss],
                          return_merged=False,
                          return_numpy=True)
      loss_vals.append(loss_v[0][0][0])
  return loss_vals


if __name__ == "__main__":
  # place = paddle.CPUPlace()
  place = paddle.CUDAPlace(0)

  loop_num = 10
  feed = RandFeedData(10)

  loss_f = Run(place, loop_num, feed, use_cinn=False)
  loss_t = Run(place, loop_num, feed, use_cinn=True)

  print("Paddle: ", loss_f)
  print("CINN: ", loss_t)

  print(np.max(np.fabs(np.asarray(loss_f) - np.asarray(loss_t))))
  assert(np.allclose(loss_t, loss_f, atol=1e-5))

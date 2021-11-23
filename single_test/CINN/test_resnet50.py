import numpy as np
import six, sys
import paddle

paddle.enable_static()

def reader(limit):
    for i in range(limit):
        yield np.random.randint(0, 256, size=[1, 3, 224, 224]).astype('float32'), \
              np.random.randint(0, 1000, size=[1]).astype('int64')

def RandFeedData(loop_num=10):
  feed = []
  data = reader(loop_num)
  for i in range(loop_num):
    x, y = next(data)
    feed.append({'image' : x, 'label' : y})
  return feed

def BuildProgram(main_program, startup_program):
  with paddle.static.program_guard(main_program, startup_program):
    image = paddle.static.data(name='image', shape=[1, 3, 224, 224], dtype='float32')
    label = paddle.static.data(name='label', shape=[1], dtype='int64')
    model = paddle.vision.models.resnet50()
    prediction = model(image)
    loss = paddle.nn.functional.cross_entropy(input=prediction, label=label)
    loss = paddle.mean(loss)
    adam = paddle.optimizer.Adam(learning_rate=0.001)
    adam.minimize(loss)
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
  feed = RandFeedData(10)

  # loss_f = Run(place, loop_num, feed, use_cinn=False)
  loss_t = Run(place, loop_num, feed, use_cinn=True)

  # print(np.max(np.fabs(np.asarray(loss_f) - np.asarray(loss_t))))

import sys
import paddle
import numpy as np
import time
from numba import cuda
from paddle.fluid import layers

loopNum = 1000


##### static mode
paddle.enable_static()

np.random.seed(int(time.time()))
cuda.select_device(0)

def cond(i, loopNum, x, y, out):
  return i < loopNum

def body(i, loopNum, x, y, out):
  paddle.increment(i)
  out = layers.matmul(pd_x, pd_y)
  return [i, loopNum, x, y, out]

print("Please Input matrix Dim [m, k, n]:")
m, k, n = map(int, input().split())

if m % 8 == 0 and k % 8 == 0 and n % 8 == 0 :
  print("Using tensor core")
else:
  print("Not using tensor core")

# generate number
x = np.random.uniform(0.1, 1, (m, k))
y = np.random.uniform(0.1, 1, (k, n))

i = layers.fill_constant(shape=[1], dtype='int64', value=0)
loop = layers.fill_constant(shape=[1], dtype='int64', value=loopNum)
pd_x = layers.data(name='x', shape=[m, k], dtype='float16', stop_gradient=True)
pd_y = layers.data(name='y', shape=[k, n], dtype='float16', stop_gradient=True)

out_pd = layers.matmul(pd_x, pd_y)
i, loop, pd_x, pd_y, out_pd = layers.while_loop(cond, body, [i, loop, pd_x, pd_y, out_pd])
# out_pd = layers.matmul(pd_x, pd_y)

# create an executor using GPU
exe = paddle.fluid.Executor(paddle.CUDAPlace(0))
exe.run(paddle.fluid.default_startup_program())

# Execute
t1 = time.time()
res, = exe.run(paddle.fluid.default_main_program(),
              feed={'x':x, 'y':y},
              fetch_list=[out_pd],
              return_numpy=True)
exe.close()
t2 = time.time()
time_paddle = t2 - t1
print("[{}, {}, {}] Cost: {} s".format(m, k, n, time_paddle))

print("******************************************")
cuda.close




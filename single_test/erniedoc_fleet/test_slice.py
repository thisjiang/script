import paddle
import tensorflow as tf
import torch
import numpy as np
import time
from numba import cuda

loopNum = 1000

np.random.seed(int(time.time()))
cuda.select_device(0)

# initial input value
'''
x_in = [[1,2,3,4],[5,6,7,8],]
x_axes = [0,1]
x_starts = [1,0]
x_ends = [2,3]
'''
if False:
  x_in = np.random.uniform(0, 10, (4, 12, 1152, 512))
  x_axes = [0, 1, 2, 3]
  x_starts = [0, 0, 1, 0]
  x_ends = [10000000, 10000000, 10000000, 10000000]
else:
  x_in = np.random.uniform(0, 10, (4, 12, 512, 1151))
  x_axes = [0, 1, 2, 3]
  x_starts = [0, 0, 0, 0]
  x_ends = [10000000, 10000000, 10000000, 640]

# paddle
pd_in = paddle.to_tensor(data=x_in, dtype='float16', place=paddle.CUDAPlace(0), stop_gradient=False)
pd_starts = paddle.to_tensor(data=x_starts, dtype='int32', place=paddle.CUDAPlace(0))
pd_ends = paddle.to_tensor(data=x_ends, dtype='int32', place=paddle.CUDAPlace(0))
out_pd = paddle.slice(pd_in, x_axes, pd_starts, pd_ends)
pd_res = out_pd.numpy()
out_pd.backward()
pd_din = pd_in.grad

if x_in.size < 100:
  print("input:")
  for i in x_in:
    print(i, end=' ')
  print("")
  print("result:")
  for i in pd_res:
    print(i, end=' ')
  print("")
  print("gradient:")
  for i in pd_din:
    print(i, end=' ')
  print("")

# time
time_paddle = 0.0

# paddle
t1 = time.time()
for i in range(loopNum):
    out_pd = paddle.slice(pd_in, x_axes, x_starts, x_ends)
    out_pd.backward()
out_pd.numpy()
t2 = time.time()
time_paddle = t2 - t1
print("paddle: " + str(time_paddle) + " s")
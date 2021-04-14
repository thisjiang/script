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
  x_sizes = [4, 12, 1151, 512]
else:
  x_in = np.random.uniform(0, 10, (4, 12, 512, 1151))
  x_axes = [0, 1, 2, 3]
  x_starts = [0, 0, 0, 0]
  x_ends = [10000000, 10000000, 10000000, 640]
  x_sizes = [4, 12, 512, 640]

# paddle
pd_in = paddle.to_tensor(data=x_in, dtype='float16', place=paddle.CUDAPlace(0), stop_gradient=False)
pd_starts = paddle.to_tensor(data=x_starts, dtype='int32', place=paddle.CUDAPlace(0))
pd_ends = paddle.to_tensor(data=x_ends, dtype='int32', place=paddle.CUDAPlace(0))
out_pd = paddle.slice(pd_in, x_axes, pd_starts, pd_ends)
pd_res = out_pd.numpy()
out_pd.backward()
pd_dx = pd_in.grad

# tensorflow
with tf.device('/device:GPU:0'):
  tf_x = tf.constant(x_in, dtype=tf.float16)
  tf_starts = tf.constant(x_starts, dtype=tf.int32)
  tf_sizes = tf.constant(x_sizes, dtype=tf.int32)

with tf.GradientTape() as g:
    g.watch(tf_x)
    out_tf = tf.slice(tf_x, tf_starts, tf_sizes)
tf_res = out_tf.numpy()
tf_list = g.gradient(out_tf, [tf_x])
tf_dx = tf_list[0].numpy()

# check result
tf_dxerr = np.max(np.abs(pd_dx - tf_dx))
print("Paddle Dx diff Tensorflow: {}".format(tf_dxerr))
if (tf_dxerr > 1e-4):
  print("Tensorflow Compare ERROR")
  exit(0)

if x_in.size < 100:
  print("input:")
  for i in x_in:
    print(i, end=' ')
  print("")
  print("paddle result:")
  for i in pd_res:
    print(i, end=' ')
  print("")
  print("tensorflow result:")
  for i in tf_res:
    print(i, end=' ')
  print("")
  print("paddl gradient:")
  for i in pd_dx:
    print(i, end=' ')
  print("")
  print("tensorflow gradient:")
  for i in tf_dx:
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

# tensorflow
t1 = time.time()
for i in range(loopNum):
    with tf.GradientTape() as g:
        g.watch(tf_x)
        out_tf = tf.slice(tf_x, tf_starts, tf_sizes)
        g.gradient(out_tf, [tf_x])
out_tf.numpy()
t2 = time.time()
time_tf = t2 - t1
print("tensorflow: " + str(time_tf) + " s")
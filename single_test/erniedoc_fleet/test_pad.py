import paddle
import tensorflow as tf
import torch
import numpy as np
import time
from numba import cuda

loopNum = 1000

np.random.seed(int(time.time()))
cuda.select_device(0)

# padding
if False:
  x = np.random.uniform(0, 10, (4, 12, 1151, 512))
  pad = [0, 0, 1, 0]
  pad_tf = [[0, 0], [0, 0], [1, 0], [0, 0]]
else:
  x = np.random.uniform(0, 10, (4, 12, 512, 640))
  pad = [0, 511, 0, 0]
  pad_tf = [[0, 0], [0, 0], [0, 0], [0, 511]]

mode = 'constant'
pad_value = 0

# paddle
pd_in = paddle.to_tensor(data=x, dtype='float16', place=paddle.CUDAPlace(0))
out_pd = paddle.nn.functional.pad(pd_in, pad, mode=mode, value=pad_value)
pd_res = out_pd.numpy()

#tensorflow
with tf.device('/device:GPU:0'):
    tf_x = tf.constant(x, dtype=tf.float16)
    tf_pad = tf.constant(pad_tf, dtype=tf.int32)
out_tf = tf.compat.v1.pad(tf_x, tf_pad, mode=mode, constant_values=pad_value)
tf_res = out_tf.numpy()

#pytorch
th_x = torch.tensor(x, dtype=torch.float16, device=torch.device('cuda:0'))
out_th = torch.nn.functional.pad(th_x, pad=pad, mode=mode, value=pad_value)
th_res = out_th.cpu().numpy()


if x.size < 100:
  print("input:")
  for i in x:
    print(i, end=' ')
  print("")
  print("paddle:")
  for i in pd_res:
    print(i, end=' ')
  print("")
  print("tensorflow:")
  for i in tf_res:
    print(i, end=' ')
  print("")
  print("pytorch:")
  for i in th_res:
    print(i, end=' ')
  print("")

tf_err = np.max(np.abs(pd_res - tf_res))
print("Paddle diff Tensorflow: {}".format(tf_err))
if (tf_err > 1e-4):
    print("Tensorflow Compare ERROR")
    exit(0)

th_err = np.max(np.abs(pd_res - th_res))
print("Paddle diff Pytorch: {}".format(th_err))
if (th_err > 1e-4):
    print("Pytorch Compare ERROR")
    # exit(0)

# time
time_paddle = 0.0
time_tf = 0.0
time_torch = 0.0

# time
t1 = time.time()
for i in range(loopNum):
    out_pd = paddle.nn.functional.pad(pd_in, pad, mode, value=pad_value)
out_pd.numpy()
t2 = time.time()
time_paddle = t2 - t1
print("paddle cost: " + str(time_paddle) + " s")

t1 = time.time()
for i in range(loopNum):
    out_tf = tf.compat.v1.pad(tf_x, tf_pad, mode=mode, constant_values=pad_value)
out_tf.numpy()
t2 = time.time()
time_tf = t2 - t1
print("tensorflow cost: " + str(time_tf) + " s")

t1 = time.time()
for i in range(loopNum):
    out_th = torch.nn.functional.pad(th_x, pad=pad, mode=mode, value=pad_value)
torch.cuda.synchronize()
t2 = time.time()
time_torch = t2 - t1
print("pytorch cost: " + str(time_torch) + " s")
print("******************************************")
cuda.close
torch.cuda.empty_cache()



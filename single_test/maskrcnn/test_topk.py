import sys
import paddle
import paddle.fluid.core as core
import tensorflow as tf
import torch
import numpy as np
import time
from numba import cuda

loopNum = 100

np.random.seed(int(time.time()))
cuda.select_device(0)


x = np.random.rand(1, 150402)
# x = np.random.rand(20, 2)
k = 1
axis = 0

# paddle
pd_x = paddle.to_tensor(data=x, dtype='float32', place=paddle.CUDAPlace(0))
value_pd, ind_pd = paddle.topk(pd_x, k, axis=axis, sorted=True)
pd_res = value_pd.numpy()

with tf.device('/device:GPU:0'):
    tf_x = tf.constant(x, dtype=tf.float32)
    tf_x = tf.transpose(tf_x, perm=[1, 0])
value_tf, ind_tf = tf.math.top_k(tf_x, k=k, sorted=True)
value_tf = tf.transpose(value_tf, perm=[1, 0])
tf_res = value_tf.numpy()

#pytorch
th_x = torch.tensor(x, dtype=torch.float32, device=torch.device('cuda:0'))
value_th, ind_th = torch.topk(th_x, k, axis=axis, sorted=True)
th_res = value_th.cpu().numpy()


if x.size <= 100:
    print('x')
    print(x.shape)
    for i in x:
        print(i, end = ' ')
    print('')
    print('Paddle')
    print(pd_res.shape)
    for id in pd_res:
        print(id, end = ' ')
    print('')
    print('Tensorflow')
    print(tf_res.shape)
    for id in tf_res:
        print(id, end = ' ')
    print('')
    print('Pytoch')
    print(th_res.shape)
    for id in th_res:
        print(id, end = ' ')
    print('')

if (pd_res != tf_res).any():
    print("Tensorflow Check Error")
    quit()

if (pd_res != th_res).any():
    print("Pytorch Check Error")
    quit()


# time
time_paddle = 0.0
time_tf = 0.0
time_torch = 0.0

t1 = time.time()
for i in range(loopNum):
    value_pd, ind_pd = paddle.topk(pd_x, k, sorted=True)
value_pd.numpy()
t2 = time.time()
time_paddle = t2 - t1
print("paddle cost: " + str(time_paddle) + " s")

t1 = time.time()
for i in range(loopNum):
    value_tf, ind_tf = tf.math.top_k(tf_x, k=k, sorted=True)
value_tf.numpy()
t2 = time.time()
time_tf = t2 - t1
print("tensorflow cost: " + str(time_tf) + " s")

t1 = time.time()
for i in range(loopNum):
    value_th, ind_th = torch.topk(th_x, k, dim=0, sorted=True)
torch.cuda.synchronize()
t2 = time.time()
time_torch = t2 - t1
print("pytorch cost: " + str(time_torch) + " s")
print("******************************************")
cuda.close
torch.cuda.empty_cache()
import paddle
import paddle.fluid.core as core
import tensorflow as tf
import torch
import numpy as np
import time

np.random.seed(int(time.time()))

x = np.random.rand(2, 3, 100000)
y = np.random.rand(100000)

#paddle
pd_x = paddle.to_tensor(data=x, place=paddle.CUDAPlace(0))
pd_y = paddle.to_tensor(data=y, place=paddle.CUDAPlace(0))

out_pd = paddle.fluid.layers.elementwise_max(pd_x, pd_y)
pd_res = out_pd.numpy()
#tensorflow
with tf.device('/device:GPU:0'):
  tf_x = tf.constant(x)
  tf_y = tf.constant(y)

out_tf = tf.math.maximum(tf_x, tf_y)
tf_res = out_tf.numpy()
#pytorch
th_x = torch.tensor(x, device=torch.device('cuda:0'))
th_y = torch.tensor(y, device=torch.device('cuda:0'))

out_th = torch.maximum(th_x, th_y)
th_res = out_th.cpu().numpy()

if (pd_res != tf_res).any() or (pd_res != th_res).any():
    print("Compare ERROR")
    quit()

if x.size <= 200 and y.size <= 200:
    print('x')
    for i in x:
        print(i, end = ' ')
    print('')
    print('y')
    for i in  y:
        print(i, end = ' ')
    print('')
    print('Paddle')
    for id in pd_res:
        print(id, end = ' ')
    print('')
    print('Tensorflow')
    for id in tf_res:
        print(id, end = ' ')
    print('')
    print('Pytoch')
    for id in th_res:
        print(id, end = ' ')
    print('')


# time
time_paddle = 0.0
time_tf = 0.0
time_torch = 0.0

t1 = time.time()
for i in range(1000):
    out_pd = paddle.fluid.layers.elementwise_max(pd_x, pd_y)
pd_res = out_pd.numpy()
t2 = time.time()
time_paddle = t2 - t1
print("paddle: " + str(time_paddle) + " ms")

t1 = time.time()
for i in range(1000):
    out_tf = tf.math.maximum(tf_x, tf_y)
tf_res = out_tf.numpy()
t2 = time.time()
time_tf = t2 - t1
print("tensorflow: " + str(time_tf) + " ms")

t1 = time.time()
for i in range(1000):
    out_th = torch.maximum(th_x, th_y)
th_res = out_th.cpu().numpy()
t2 = time.time()
time_torch = t2 - t1
print("pytorch: " + str(time_torch) + " ms")
import paddle
import paddle.fluid.core as core
import tensorflow as tf
import torch
import numpy as np
import time

np.random.seed(int(time.time()))

LOOPNUM = 1000

x = np.random.rand(2, 10, 100, 100)
y = np.random.rand(100, 100)

#paddle
pd_x = paddle.to_tensor(data=x, place=paddle.CUDAPlace(0), stop_gradient=False)
pd_y = paddle.to_tensor(data=y, place=paddle.CUDAPlace(0), stop_gradient=False)

out_pd = paddle.fluid.layers.elementwise_max(pd_x, pd_y)
pd_res = out_pd.numpy()
out_pd.backward()

#tensorflow
with tf.device('/device:GPU:0'):
  tf_x = tf.constant(x)
  tf_y = tf.constant(y)

with tf.GradientTape() as g:
    g.watch(tf_x)
    g.watch(tf_y)
    out_tf = tf.math.maximum(tf_x, tf_y)
tf_res = out_tf.numpy()
g.gradient(out_tf, [tf_x, tf_y])

#pytorch
th_x = torch.tensor(x, device=torch.device('cuda:0'), requires_grad=True)
th_y = torch.tensor(y, device=torch.device('cuda:0'), requires_grad=True)

out_th = torch.maximum(th_x, th_y)
th_res = out_th.cpu().detach().numpy()
out_th.backward(out_th.data)

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
for i in range(LOOPNUM):
    out_pd = paddle.fluid.layers.elementwise_max(pd_x, pd_y)
    out_pd.backward()
pd_res = out_pd.numpy()
t2 = time.time()
time_paddle = t2 - t1
print("paddle: " + str(time_paddle) + " ms")

t1 = time.time()
for i in range(LOOPNUM):
    with tf.GradientTape() as g:
        out_tf = tf.math.maximum(tf_x, tf_y)
        g.gradient(out_tf, [tf_x, tf_y])
tf_res = out_tf.numpy()
t2 = time.time()
time_tf = t2 - t1
print("tensorflow: " + str(time_tf) + " ms")

t1 = time.time()
for i in range(LOOPNUM):
    out_th = torch.maximum(th_x, th_y)
    out_th.backward(out_th.data)
th_res = out_th.cpu().detach().numpy()
t2 = time.time()
time_torch = t2 - t1
print("pytorch: " + str(time_torch) + " ms")

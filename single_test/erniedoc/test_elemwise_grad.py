import paddle
import paddle.fluid.core as core
import tensorflow as tf
import torch
import numpy as np
import time
from numba import cuda

np.random.seed(int(time.time()))
cuda.select_device(0)

LOOPNUM = 1000

x = np.random.uniform(0.1, 0.9, (100, 2048, 7, 7))
y = np.random.uniform(0.1, 0.9, (100, 2048))
axis = 0

#paddle
pd_x = paddle.to_tensor(data=x, dtype='float32', place=paddle.CUDAPlace(0), stop_gradient=False)
pd_y = paddle.to_tensor(data=y, dtype='float32', place=paddle.CUDAPlace(0), stop_gradient=False)

out_pd = paddle.fluid.layers.elementwise_div(pd_x, pd_y, axis=axis)
pd_res = out_pd.numpy()
out_pd.backward()
pd_dx = pd_x.grad
pd_dy = pd_y.grad

#tensorflow
with tf.device('/device:GPU:0'):
  tf_x = tf.constant(x, dtype=tf.float32)
  tf_y = tf.constant(y, dtype=tf.float32)

with tf.GradientTape() as g:
    g.watch(tf_x)
    g.watch(tf_y)
    out_tf = tf.math.divide(tf_x, tf_y)
tf_res = out_tf.numpy()
tf_list = g.gradient(out_tf, [tf_x, tf_y])
tf_dx = tf_list[0].numpy()
tf_dy = tf_list[1].numpy()

#pytorch
th_x = torch.tensor(x, dtype=torch.float32, device=torch.device('cuda:0'), requires_grad=True)
th_y = torch.tensor(y, dtype=torch.float32, device=torch.device('cuda:0'), requires_grad=True)

out_th = torch.div(th_x, th_y)
th_res = out_th.cpu().detach().numpy()
out_th.backward(gradient=torch.ones_like(out_th))
th_dx = th_x.grad.cpu().numpy()
th_dy = th_y.grad.cpu().numpy()

# check dx result
pd_maxdx = np.max(pd_dx)
tf_maxdx = np.max(tf_dx)
th_maxdx = np.max(th_dx)

print("Max Dx result: Paddle {} Tensorflow {} Pytorch {}"
        .format(pd_maxdx, tf_maxdx, th_maxdx))

tf_dxerr = np.max(np.abs(pd_dx - tf_dx))
print("Paddle Dx diff Tensorflow: {}".format(tf_dxerr))
if (tf_dxerr > 1e-4):
    print("Tensorflow Compare ERROR")
    #quit()

th_dxerr = np.max(np.abs(pd_dx - th_dx))
print("Paddle Dx diff Pytorch: {}".format(th_dxerr))
if (th_dxerr > 1e-4):
    print("Pytorch Compare ERROR")
    #quit()

# check dy result
pd_maxdy = np.max(pd_dy)
tf_maxdy = np.max(tf_dy)
th_maxdy = np.max(th_dy)

print("Max Dy result: Paddle {} Tensorflow {} Pytorch {}"
        .format(pd_maxdy, tf_maxdy, th_maxdy))

tf_dyerr = np.max(np.abs(pd_dy - tf_dy))
print("Paddle Dy diff Tensorflow: {}".format(tf_dyerr))
if (tf_dyerr > 1e-4):
    print("Tensorflow Compare ERROR")
    #quit()

th_dyerr = np.max(np.abs(pd_dy - th_dy))
print("Paddle Dy diff Pytorch: {}".format(th_dyerr))
if (th_dyerr > 1e-4):
    print("Pytorch Compare ERROR")
    #quit()

if x.size <= 500:
    print('X')
    print(x.shape)
    for id in x:
        print(id, end = ' ')
    print('')
    print('Paddle')
    print(pd_dx.shape)
    for id in pd_dx:
        print(id, end = ' ')
    print('')
    print('Tensorflow')
    print(tf_dx.shape)
    for id in tf_dx:
        print(id, end = ' ')
    print('')
    print('Pytoch')
    print(th_dx.shape)
    for id in th_dx:
        print(id, end = ' ')
    print('')

if y.size <= 500:
    print('Y')
    print(y.shape)
    for id in y:
        print(id, end = ' ')
    print('')
    print('Paddle')
    print(pd_dy.shape)
    for id in pd_dy:
        print(id, end = ' ')
    print('')
    print('Tensorflow')
    print(tf_dy.shape)
    for id in tf_dy:
        print(id, end = ' ')
    print('')
    print('Pytoch')
    print(th_dy.shape)
    for id in th_dy:
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
out_pd.numpy()
t2 = time.time()
time_paddle = t2 - t1
print("paddle: " + str(time_paddle) + " s")

t1 = time.time()
for i in range(LOOPNUM):
    with tf.GradientTape() as g:
        out_tf = tf.math.maximum(tf_x, tf_y)
        g.gradient(out_tf, [tf_x, tf_y])
out_tf.numpy()
t2 = time.time()
time_tf = t2 - t1
print("tensorflow: " + str(time_tf) + " s")

t1 = time.time()
for i in range(LOOPNUM):
    out_th = torch.maximum(th_x, th_y)
    out_th.backward(out_th.data)
torch.cuda.synchronize()
t2 = time.time()
time_torch = t2 - t1
print("pytorch: " + str(time_torch) + " s")
print("******************************************")
cuda.close
torch.cuda.empty_cache()

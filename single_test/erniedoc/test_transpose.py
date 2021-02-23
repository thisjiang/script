import paddle
import tensorflow as tf
import torch
import numpy as np
import time

loopNum = 1000

np.random.seed(int(time.time()))
x = np.random.rand(1, 458752, 48)

#paddle
pd_x = paddle.to_tensor(data=x, dtype='float16', place=paddle.CUDAPlace(0))
out_pd = paddle.transpose(pd_x, perm=[0, 2, 1])
pd_res = out_pd.numpy()

#tensorflow
with tf.device('/device:GPU:0'):
  tf_x = tf.constant(x, dtype=tf.float16)
out_tf = tf.transpose(tf_x, perm=[0, 2, 1])
tf_res = out_tf.numpy()

#pytorch
th_x = torch.tensor(x, dtype=torch.float16, device=torch.device('cuda:0'))
out_th = torch.transpose(th_x, 1, 2)
th_res = out_th.cpu().numpy()

# check result
if (pd_res != tf_res).any():
  print("Tensorflow Compare ERROR")
  quit()
'''
if (pd_res != th_res).any():
  print("Pytorch Compare ERROR")
  quit()
'''
if x.size <= 500:
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


# time
time_paddle = 0.0
time_tf = 0.0
time_torch = 0.0

t1 = time.time()
for i in range(loopNum):
    out_pd = paddle.transpose(pd_x, perm=[0, 2, 1])
pd_res = out_pd.numpy()
t2 = time.time()
time_paddle = t2 - t1
print("paddle: " + str(time_paddle) + " ms")

t1 = time.time()
for i in range(loopNum):
    out_tf = tf.transpose(tf_x, perm=[0, 2, 1])
tf_res = out_tf.numpy()
t2 = time.time()
time_tf = t2 - t1
print("tensorflow: " + str(time_tf) + " ms")

t1 = time.time()
for i in range(loopNum):
    out_th = torch.transpose(th_x, 1, 2)
th_res = out_th.cpu().numpy()
t2 = time.time()
time_torch = t2 - t1
print("pytorch: " + str(time_torch) + " ms")
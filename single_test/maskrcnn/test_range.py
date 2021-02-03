import paddle
import tensorflow as tf
import torch
import numpy as np
import time

time_paddle = 0.0
time_tf = 0.0
time_torch = 0.0

start = 0
end = 100000
step = 1
dtype = 'int64'
th_dtype = torch.int64
# paddle
out_pd = paddle.fluid.layers.range(start, end, step, dtype)
pd_res = out_pd.numpy()
# tensorflow
out_tf = tf.range(start=start, limit=end, delta=step, dtype=dtype)
tf_res = out_tf.numpy()
# pytorch
out_th = torch.arange(start=start, end=end, step=step, dtype=th_dtype)
th_res = out_th.numpy()

if (pd_res != tf_res).any() or (pd_res != th_res).any():
    print("Compare ERROR")
    quit()

size = (end - start) / step
if size <= 200:
    print("start {} end {} step {} size {} :".format(start, end, step, size))
    print("Paddle")
    for id in pd_res:
        print(id, end = ' ')
    print('')
    print('Tensorflow')
    for id in tf_res:
        print(id, end = ' ')
    print('')
    print('pytorch')
    for id in th_res:
        print(id, end = ' ')
    print('')

t1 = time.time()
for i in range(1000):
    out_pd = paddle.fluid.layers.range(start, end, step, dtype)
out_pd.numpy()
t2 = time.time()
time_paddle = t2 - t1
print("paddle: " + str(time_paddle) + " ms")

t1 = time.time()
for i in range(1000):
    out_tf = tf.range(start=start, limit=end, delta=step, dtype=dtype)
out_tf.numpy()
t2 = time.time()
time_tf = t2 - t1
print("tensorflow: " + str(time_tf) + " ms")

t1 = time.time()
for i in range(1000):
    out_th = torch.arange(start=start, end=end, step=step, dtype=th_dtype)
out_th.numpy()
t2 = time.time()
time_torch = t2 - t1
print("pytorch: " + str(time_torch) + " ms")
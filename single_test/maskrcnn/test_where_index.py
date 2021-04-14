import paddle
import paddle.fluid.core as core
import tensorflow as tf
import torch
import numpy as np
import time

time_paddle = 0.0
time_tf = 0.0
time_torch = 0.0

condition = np.random.randint(0, 2, [100, 100, 100])

# paddle
pd_condition = paddle.to_tensor(data=condition, dtype='int32', place=paddle.CUDAPlace(0))
out_index = core.ops.where_index(pd_condition)
pd_res = out_index.numpy()
# tensorflow
with tf.device('/device:GPU:0'):
    tf_condition = tf.constant(condition, dtype='int32')
tf_out = tf.where(tf_condition)
tf_res = tf_out.numpy()
# pytorch
th_condition = torch.tensor(condition, dtype=torch.int32, device=torch.device('cuda:0'))
th_out = torch.nonzero(torch.tensor(th_condition))
th_res = th_out.cpu().numpy()

if (pd_res != tf_res).any() or (pd_res != th_res).any():
    print("Compare ERROR")
    quit()

if condition.size <= 200:
    for cond in condition:
        print(cond, end = ' ')
    print('')
    for id in out_index.numpy():
        print(id, end = ' ')
    print('')
    for id in tf_out.numpy():
        print(id, end = ' ')
    print('')
    for id in th_out.numpy():
        print(id, end = ' ')
    print('')


t1 = time.time()
for i in range(100):
    out_index = core.ops.where_index(pd_condition)
out_index.numpy()
t2 = time.time()
time_paddle = t2 - t1
print("paddle: " + str(time_paddle) + " ms")

t1 = time.time()
for i in range(100):
    tf_out = tf.where(tf_condition)
tf_out.numpy()
t2 = time.time()
time_tf = t2 - t1
print("tensorflow: " + str(time_tf) + " ms")

t1 = time.time()
for i in range(100):
    th_out = torch.nonzero(th_condition)
th_out.cpu().numpy()
t2 = time.time()
time_torch = t2 - t1
print("pytorch: " + str(time_torch) + " ms")

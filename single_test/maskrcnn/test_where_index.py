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
tensor_data = paddle.to_tensor(data=condition, place=paddle.CUDAPlace(0))
# paddle
out_index = core.ops.where_index(tensor_data)
pd_res = out_index.numpy()
# tensorflow
tf_out = tf.where(condition)
tf_res = tf_out.numpy()
# pytorch
th_out = torch.nonzero(torch.tensor(condition))
th_res = th_out.numpy()

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
    out_index = core.ops.where_index(tensor_data)
out_index.numpy()
t2 = time.time()
time_paddle = t2 - t1
print("paddle: " + str(time_paddle) + " ms")

t1 = time.time()
for i in range(100):
    tf_out = tf.where(condition)
tf_out.numpy()
t2 = time.time()
time_tf = t2 - t1
print("tensorflow: " + str(time_tf) + " ms")

t1 = time.time()
for i in range(100):
    th_out = torch.nonzero(torch.tensor(condition))
th_out.numpy()
t2 = time.time()
time_torch = t2 - t1
print("pytorch: " + str(time_torch) + " ms")

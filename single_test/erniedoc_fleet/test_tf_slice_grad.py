import paddle
import tensorflow as tf
from tensorflow.python.ops import array_ops
import numpy as np
import time
from numba import cuda

np.random.seed(int(time.time()))
cuda.select_device(0)

inputs = np.random.uniform(0, 10, (4, 12, 1152, 512))
outputs = np.random.uniform(0, 10, (4, 12, 1151, 512))
grad = np.random.uniform(0, 10, (4, 12, 1151, 512))

# tensorflow
with tf.device('/device:GPU:0'):
  tf_inputs = tf.constant(inputs, dtype=tf.float32)
  tf_outputs = tf.constant(outputs, dtype=tf.float32)
  tf_grad = tf.constant(grad, dtype=tf.float32)

input_vec = tf_inputs[0]
begin_vec = tf_inputs[1]
input_rank = array_ops.rank(input_vec)
slice_size = array_ops.shape(tf_outputs[0])
shape = array_ops.stack([input_rank, 1])
before_pad = array_ops.reshape(begin_vec, shape)
after_pad = array_ops.reshape(
    array_ops.shape(input_vec) - slice_size - begin_vec, shape)
paddings = array_ops.concat([before_pad, after_pad], 1)
res = array_ops.pad(tf_grad, paddings)
res.numpy()

# paddle
pd_inputs = paddle.to_tensor(data=inputs, dtype='float16', place=paddle.CUDAPlace(0))
pd_outputs = paddle.to_tensor(data=outputs, dtype='float16', place=paddle.CUDAPlace(0))
pd_grad = paddle.to_tensor(data=grad, dtype='float16', place=paddle.CUDAPlace(0))

input_vec_pd = pd_inputs[0]
begin_vec_pd = pd_inputs[1]

input_rank_pd = paddle.rank(input_vec_pd)
slice_size_pd = paddle.shape(pd_outputs[0])
shape_pd = paddle.stack([input_rank_pd, 1])
before_pad_pd = paddle.reshape(begin_vec_pd, shape_pd)
after_pad_pd = paddle.reshape(
    paddle.shape(input_vec_pd) - slice_size_pd - begin_vec_pd, shape_pd)
paddings_pd = paddle.concat([before_pad_pd, after_pad_pd], 1)
res_pd = paddle.pad(pd_grad, paddings_pd)
res_pd.numpy()

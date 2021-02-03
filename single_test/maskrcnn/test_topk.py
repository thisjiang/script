import paddle
import paddle.fluid.core as core
import numpy as np
import time

time_topk = 0.0
time_max = 0.0
data = np.random.rand(20, 242991)
tensor_data = paddle.to_tensor(data=data, place=paddle.CUDAPlace(0))
match_max = paddle.max(tensor_data, axis=0)
match_max.numpy()

matched_vals, matches = core.ops.top_k_v2(tensor_data, 'k', 1, 'axis', 0)
matched_vals.numpy()

t1 = time.time()
for i in range(100):
    matched_vals, matches = core.ops.top_k_v2(tensor_data, 'k', 1, 'axis', 0)
matched_vals.numpy()
t2 = time.time()
time_topk = t2 - t1

t3 = time.time()
for i in range(100):
    match_max = paddle.max(tensor_data, axis=0)
match_max.numpy()
t4 = time.time()
time_max = t4 - t3

print(time_topk)
print(time_max)
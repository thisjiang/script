import paddle
import numpy as np
import time

loopNum = 1000

np.random.seed(int(time.time()))

x = np.random.uniform(0, 0.01, (5, 6, 10))
# x = np.random.randint(0, 30, (10, 10))
dim = [0, 1, 2]
keep_dim = False

# numpy
np_res = x.sum(axis = None)
np_dx = np.gradient(np_res)

# float32
pd_x = paddle.to_tensor(data=x, dtype='float32', place=paddle.CUDAPlace(0), stop_gradient=False)
out_pd = paddle.fluid.layers.reduce_sum(input=pd_x, dim=dim, keep_dim=keep_dim)
# out_pd = pd_x.sum(axis = None)
fp_res = out_pd.numpy()

out_pd.backward()
fp_dx = pd_x.grad

# float16
pd_x = paddle.to_tensor(data=x, dtype='float16', place=paddle.CUDAPlace(0), stop_gradient=False)
out_pd = paddle.fluid.layers.reduce_sum(input=pd_x, dim=dim, keep_dim=keep_dim)
# out_pd = pd_x.sum(axis = None)
out_pd = paddle.cast(out_pd, 'float32')
fp16_res = out_pd.numpy()

out_pd.backward()
fp16_dx = pd_x.grad

# 
print("numpy: {}".format(np_res))
print("float32: {}, acc: {}".format(fp_res, np_res - fp_res))
print("float16: {}, acc: {}".format(fp16_res, np_res - fp16_res))
''''
print("numpy grad")
print(np_dx)

print("float32 grad")
print(fp_dx)

print("float16 grad")
print(fp16_dx)
'''

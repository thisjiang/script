import sys
import paddle
import numpy as np
import time
from numba import cuda
from paddle.fluid import layers

loopNum = 1000
padding = False

np.random.seed(int(time.time()))
cuda.select_device(0)

def ceil_to_8(num):
  if num % 8 == 0:
    return num
  return (num + 7) & ~7

while True:
  print("Please Input matrix Dim [m, k, n]:")
  m, k, n = map(int, input().split())

  if m % 8 == 0 and k % 8 == 0 and n % 8 == 0 :
    print("Using tensor core")
  else:
    print("Not using tensor core")

  # generate number
  x = np.random.uniform(0.1, 1, (m, k))
  y = np.random.uniform(0.1, 1, (k, n))

  # dygraph mode
  pd_x = paddle.to_tensor(data=x, dtype='float16', place=paddle.CUDAPlace(0), stop_gradient=False)
  pd_y = paddle.to_tensor(data=y, dtype='float16', place=paddle.CUDAPlace(0), stop_gradient=False)

  # padding layer
  if padding:
    print("Using Padding")
    old_x_shape = pd_x.shape
    old_y_shape = pd_y.shape

    pad_x_right = ceil_to_8(old_x_shape[-2]) - old_x_shape[-2]
    pad_x_bottom = ceil_to_8(old_x_shape[-1]) - old_x_shape[-1]
    pad_y_right = ceil_to_8(old_y_shape[-2]) - old_y_shape[-2]
    pad_y_bottom = ceil_to_8(old_y_shape[-1]) - old_y_shape[-1]

    pad_x = paddle.nn.Pad2D([0, pad_x_right, 0, pad_x_bottom], mode='constant')
    pad_y = paddle.nn.Pad2D([0, pad_y_right, 0, pad_y_bottom], mode='constant')

    pd_x = pad_x(pd_x)
    pd_y = pad_y(pd_y)

    print(pd_x.shape)
    print(pd_y.shape)

  # forward
  out_pd = layers.matmul(pd_x, pd_y)
  pd_res = out_pd.numpy()

  # backward
  out_pd.backward(retain_graph=True)
  pd_dx = pd_x.grad
  pd_dy = pd_y.grad

  if x.size <= 0:
    print('x')
    print(x.shape)
    for id in x:
        print(id, end = ' ')
    print('')
    print('y')
    print(y.shape)
    for id in y:
        print(id, end = ' ')
    print('')
    print('result')
    print(pd_res.shape)
    for id in pd_res:
        print(id, end = ' ')
    print('')
    '''
    print('dx')
    print(pd_dx.shape)
    for id in pd_dx:
        print(id, end = ' ')
    print('')
    print('dy')
    print(pd_dy.shape)
    for id in pd_dy:
        print(id, end = ' ')
    print('')
    '''

  # time
  t1 = time.time()
  for i in range(loopNum):
    out_pd = layers.matmul(pd_x, pd_y)
    out_pd.backward(retain_graph=True)
  out_pd.numpy()
  t2 = time.time()
  time_paddle = t2 - t1
  print("[{}, {}, {}] Cost: {} s".format(m, k, n, time_paddle))

  print("******************************************")
  cuda.close


import sys
import paddle
import tensorflow as tf
import torch
import numpy as np
import time
from numba import cuda

loopNum = 1000

np.random.seed(int(time.time()))
cuda.select_device(0)

while True:
    # x = np.random.rand(512, 896, 4, 12)
    print("Please Input Forward Dim [x, y, z]:")
    N, dim, D = map(int, input().split())
    axis = int(input("axis:"))
    x = np.random.rand(N, dim, D)

    #paddle
    pd_x = paddle.to_tensor(data=x, dtype='float32', place=paddle.CUDAPlace(0))
    out_pd = paddle.fluid.layers.softmax(pd_x, axis=axis)
    pd_res = out_pd.numpy()

    #tensorflow
    with tf.device('/device:GPU:0'):
        tf_x = tf.constant(x, dtype=tf.float32)
    out_tf = tf.nn.softmax(tf_x, axis=axis)
    tf_res = out_tf.numpy()

    #pytorch
    th_x = torch.tensor(x, dtype=torch.float32, device=torch.device('cuda:0'))
    out_th = torch.nn.functional.softmax(th_x, dim=axis)
    th_res = out_th.cpu().numpy()

    # check result
    pd_maxres = np.max(pd_res)
    tf_maxres = np.max(tf_res)
    th_maxres = np.max(th_res)

    print("Shape = [{}, {}, {}], axis = {}"
            .format(N, dim, D, axis))
    print("Max result: Paddle {} Tensorflow {} Pytorch {}"
            .format(pd_maxres, tf_maxres, th_maxres))

    tf_err = np.max(np.abs(pd_res - tf_res))
    print("Paddle diff Tensorflow: {}".format(tf_err))
    if (tf_err > 1e-4):
        print("Tensorflow Compare ERROR")
        #quit()

    th_err = np.max(np.abs(pd_res - th_res))
    print("Paddle diff Pytorch: {}".format(th_err))
    if (th_err > 1e-4):
        print("Pytorch Compare ERROR")
        #quit()

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
        out_pd = paddle.fluid.layers.softmax(pd_x, axis=axis)
    out_pd.numpy()
    t2 = time.time()
    time_paddle = t2 - t1
    print("paddle cost: " + str(time_paddle) + " s")

    t1 = time.time()
    for i in range(loopNum):
        out_tf = tf.nn.softmax(tf_x, axis=axis)
    out_tf.numpy()
    t2 = time.time()
    time_tf = t2 - t1
    print("tensorflow cost: " + str(time_tf) + " s")

    t1 = time.time()
    for i in range(loopNum):
        out_th = torch.nn.functional.softmax(th_x, dim=axis)
    torch.cuda.synchronize()
    t2 = time.time()
    time_torch = t2 - t1
    print("pytorch cost: " + str(time_torch) + " s")
    print("******************************************")
    cuda.close
    torch.cuda.empty_cache()

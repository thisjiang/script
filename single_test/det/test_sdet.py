import paddle
import paddle.fluid.core as core
import tensorflow as tf
import torch
import numpy as np
import time

np.random.seed(int(time.time()))

# axis = []
# for i in range(np.random.randint(1, 5)):
#   axis.append(np.random.randint(1, 20))
# axis.append(axis[-1])
# x = np.random.random(axis).astype(np.float32)
# x = np.ones(axis, dtype=np.float32)
# x_grad = np.ones(axis[:-2], dtype=np.int)
# x = [[3, 3], [3, 3]]
x = [[0.01, 0.02], [0.03, 0.04]]
# x_grad = [1]
# '''
# pytorch
th_x = torch.tensor(x, dtype=torch.float32, device=torch.device('cuda:0'), requires_grad=True)
torch_sign, torch_det = torch.linalg.slogdet(th_x)
torch_det_res = torch_det.cpu().detach().numpy()
if len(th_x.shape) == 2:
  torch_det.backward()
elif len(th_x.shape) == 3:
  for i in range(th_x.shape[0]):
    torch_det[i].backward(retain_graph=True)
elif len(th_x.shape) == 4:
  for i in range(th_x.shape[0]):
    for j in range(th_x.shape[1]):
      torch_det[i][j].backward(retain_graph=True)
elif len(th_x.shape) == 5:
  for i in range(th_x.shape[0]):
    for j in range(th_x.shape[1]):
      for k in range(th_x.shape[2]):
        torch_det[i][j][k].backward(retain_graph=True)
else:
  assert False, "Not Support %d" % len(th_x.shape)
torch_det_grad = th_x.grad
torch_det_grad = torch_det_grad.cpu().detach().numpy()

# x_torch_grad = torch.tensor(x_grad, dtype=torch.float32, device=torch.device('cuda:0'), requires_grad=False)
# torch_grad = x_torch_grad.unsqueeze(-1).unsqueeze(-2) * th_x.inverse().conj().transpose(-2, -1)
# '''
# tensorflow
with tf.device('/device:GPU:0'):
  tf_x = tf.constant(x, dtype=tf.float32)
with tf.GradientTape() as g:
  g.watch(tf_x)
  tf_sign, tf_det = tf.linalg.slogdet(tf_x)
tf_det_grad = g.gradient(tf_det, tf_x)
tf_det_grad = tf_det_grad.numpy()


# with tf.device('/device:GPU:0'):
#   x_tf_grad = tf.constant(x_grad, dtype=tf.float32)
# tf_grad = tf.reshape(x_tf_grad, tf.concat([tf_det.shape, [1, 1]], 0)) * tf.linalg.inv(tf_x, adjoint=True)
# '''

pd_x = paddle.to_tensor(data=x, dtype='float32', place=paddle.CPUPlace(), stop_gradient=False)
pd_sign, pd_det = paddle.linalg.slogdet(pd_x)
pd_det.backward()
pd_det_grad = pd_x.grad
pd_det_grad = pd_det_grad.numpy()

# perm = [i for i in range(len(pd_x.shape))]
# if len(perm) >= 2:
#   perm[-1] = len(perm) - 2
#   perm[-2] = len(perm) - 1
# x_pd_grad = paddle.to_tensor(data=x_grad, dtype='float32', place=paddle.CUDAPlace(0), stop_gradient=True)
# pd_grad = x_pd_grad.unsqueeze([-1, -2]) * paddle.conj(paddle.inverse(pd_x)).transpose(perm)

# '''
print("torch det origin")
print(torch_det_grad)
# print("torch python")
# print(torch_grad)

print("\n")
print("tensorflow det origin")
print(tf_det_grad)
# print("tensorflow python")
# print(tf_grad)

print("\n")
print("paddle det origin")
print(pd_det_grad)
# print("paddle python")
# print(pd_grad)
# '''

def checkSame(grad_a, grad_b):
  assert list(grad_a.shape) == list(grad_b.shape),"Shape Different! {} != {}".format(grad_a.shape, grad_b.shape)
  grad_a_flatten = grad_a.flatten()
  grad_b_flatten = grad_b.flatten()
  for i in range(len(grad_a_flatten)):
    err = np.fabs(grad_a_flatten[i] - grad_b_flatten[i])
    if grad_a_flatten[i] > 10:
      max_err = 1
    else:
      max_err = 1e-3
    assert err < max_err, "{} Error Too Big! {}:{}!={}".format(grad_a.shape, i, grad_a_flatten[i], grad_b_flatten[i])
  print("Congratulations! Check {} Successful!".format(grad_a.shape))

checkSame(torch_det_grad, pd_det_grad)
checkSame(tf_det_grad, pd_det_grad)
# '''

import paddle
import paddle.fluid.core as core
import tensorflow as tf
import torch
import numpy as np
import time

axis = []
for i in range(np.random.randint(1, 5)):
  axis.append(np.random.randint(1, 20))
axis.append(axis[-1])
# axis = [2, 2]
x = np.random.random(axis).astype(np.float64)
# x = np.ones(axis, dtype=np.float32)
# x = [[3, 3], [3, 3]]

# '''
# pytorch
th_x = torch.tensor(x, dtype=torch.float64, device=torch.device('cuda:0'), requires_grad=True)
torch_det = torch.linalg.det(th_x)
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

# torch_grad = torch_det.unsqueeze(-1).unsqueeze(-2) * th_x.inverse().transpose(-2, -1)
# '''
# tensorflow
with tf.device('/device:GPU:0'):
  tf_x = tf.constant(x, dtype=tf.float64)
with tf.GradientTape() as g:
  g.watch(tf_x)
  tf_det = tf.linalg.det(tf_x)
tf_det_grad = g.gradient(tf_det, tf_x)
tf_det_grad = tf_det_grad.numpy()

# tf_grad = tf.reshape(tf_det, tf.concat([tf_det.shape, [1, 1]], 0)) * tf.linalg.inv(tf_x, adjoint=True)
# '''

pd_x = paddle.to_tensor(data=x, dtype='float64', place=paddle.CUDAPlace(0), stop_gradient=False)
pd_det = paddle.linalg.det(pd_x)
pd_det.backward()
pd_det_grad = pd_x.grad
pd_det_grad = pd_det_grad.numpy()

# perm = [i for i in range(len(pd_x.shape))]
# perm[-1] = -2
# perm[-2] = -1
# pd_grad = pd_det.unsqueeze([-1, -2]) * paddle.inverse(pd_x).transpose(perm)

'''
print("torch det origin")
print(torch_det_grad)
print("torch python")
print(torch_grad)

print("\n")
print("tensorflow det origin")
print(tf_det_grad)
print("tensorflow python")
print(tf_grad)

print("\n")
print("paddle det origin")
print(pd_det_grad)
print("paddle python")
print(pd_grad)
'''

def checkSame(grad_a, grad_b):
  assert list(grad_a.shape) == list(grad_b.shape),"Shape Different! {} != {}".format(grad_a.shape, grad_b.shape)
  grad_a_flatten = grad_a.flatten()
  grad_b_flatten = grad_b.flatten()
  maxerr = 0.0
  for i in range(len(grad_a_flatten)):
    err = np.fabs(grad_a_flatten[i] - grad_b_flatten[i])
    maxerr = np.fmax(err, maxerr)
    assert err < 1e-3, "{} Error Too Big! {}:{}!={}".format(grad_b.shape, i, grad_a_flatten[i], grad_b_flatten[i])
  print("Max error is {}".format(maxerr))
  print("Congratulations! Check {} Successful!".format(grad_b.shape))

checkSame(torch_det_grad, pd_det_grad)
checkSame(tf_det_grad, pd_det_grad)

import paddle
import numpy as np

place = paddle.set_device("gpu:0")
x = paddle.rand([1, 32])

for i in range(1):
  print("in: ", x)
  out = paddle.nn.functional.log_softmax(x, axis=-1)
  print("changed in: ", x)
  print("output: ", out)
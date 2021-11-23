import paddle
import cinn
from cinn.frontend import *
from cinn.framework import *
from cinn import ir
from cinn import lang
from cinn.common import *
import numpy as np
import sys

builder = NetBuilder("test_transpose")
a = builder.create_input(Float(32), (2, 3), "A")
b = builder.transpose(a, axis=[1, 0])
prog = builder.build()
self.assertEqual(prog.size(), 1)
# print program
for i in range(prog.size()):
    print(prog[i])
tensor_data = [
    np.random.random([2, 3]).astype("float32")
]
result = prog.build_and_get_output(self.target, [a], tensor_data,
                                    [b])
result = result[0].numpy(self.target).reshape(-1)
print(tensor_data[0])
print(result)

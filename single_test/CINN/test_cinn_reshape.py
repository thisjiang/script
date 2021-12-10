from cinn.frontend import *
from cinn.common import *

builder = Netbuilder("test_reshape")
a = builder.create_input(Float(32), [2, 3], 'x')
# the shape of a is [2, 3]
out = builder.reshape(a, [3, 2])
# the shape of out is [3, 2]

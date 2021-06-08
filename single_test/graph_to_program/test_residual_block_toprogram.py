import paddle
import paddle.fluid as fluid
from paddle.fluid import core
from paddle.fluid.framework import IrGraph
import six
from paddle.vision.models import resnet50
from common import *

paddle.enable_static()

def residual_block_1(num):
    def conv_bn_layer(input,
                      ch_out,
                      filter_size,
                      stride,
                      padding,
                      act='relu',
                      bias_attr=False):
        tmp = fluid.layers.conv2d(
            input=input,
            filter_size=filter_size,
            num_filters=ch_out,
            stride=stride,
            padding=padding,
            act=None,
            bias_attr=bias_attr)
        return fluid.layers.batch_norm(input=tmp, act=act)

    data = fluid.layers.data(
        name='image',
        shape=[1, 1, 32, 32],
        dtype='float32',
        append_batch_size=False)
    hidden = data
    for _ in six.moves.xrange(num):
        conv = conv_bn_layer(hidden, 16, 3, 1, 1, act=None, bias_attr=True)
        short = conv_bn_layer(hidden, 16, 1, 1, 0, act=None)
        hidden = fluid.layers.elementwise_add(x=conv, y=short, act='relu')
    return hidden

def residual_block_2(hidden, quant_skip_pattern=None):
    label = fluid.layers.data(
        name='label', shape=[1, 1], dtype='int64', append_batch_size=False)
    matmul_weight = fluid.layers.create_parameter(
        shape=[1, 16, 32, 32], dtype='float32')
    hidden = fluid.layers.matmul(hidden, matmul_weight, True, True)
    if quant_skip_pattern:
        with fluid.name_scope(quant_skip_pattern):
            pool = fluid.layers.pool2d(
                input=hidden, pool_size=2, pool_type='avg', pool_stride=2)
    else:
        pool = fluid.layers.pool2d(
            input=hidden, pool_size=2, pool_type='avg', pool_stride=2)
    fc = fluid.layers.fc(input=pool, size=10)
    loss = fluid.layers.cross_entropy(input=fc, label=label)
    loss = fluid.layers.mean(loss)
    return loss

before_prog = fluid.Program()
startup = fluid.Program()

with fluid.program_guard(before_prog, startup):
    hidden = residual_block_1(2)

before_graph = IrGraph(core.Graph(before_prog.desc), for_test=False)
after_prog = before_graph.to_program()

with fluid.program_guard(before_prog, startup):
    loss = residual_block_2(hidden)
    opt = fluid.optimizer.Adam(learning_rate=0.001)
    opt.minimize(loss)

with fluid.program_guard(after_prog, startup):
    loss = residual_block_2(hidden)
    opt = fluid.optimizer.Adam(learning_rate=0.001)
    opt.minimize(loss)


# print program information
print_program(before_prog, "residual_block_before.txt")
print_program(after_prog, "residual_block_after.txt")

# print vars information
print_vars(before_prog, "residual_block_vars_before.txt")
print_vars(after_prog, "residual_block_vars_after.txt")

# print parameters information
print_parameters(before_prog, "residual_block_parameters_before.txt")
print_parameters(after_prog, "residual_block_parameters_after.txt")


import paddle
import paddle.fluid as fluid
from paddle.fluid import core
from paddle.fluid.framework import IrGraph
from common import *

paddle.enable_static()

def resnet50_loss():
    data = fluid.layers.data(name='image', shape=[3, 224, 224], dtype='float32')
    label = fluid.layers.data(name='label', shape=[1], dtype='int64')
    from paddle.vision.models import resnet50
    model = resnet50()
    prediction = model(data)
    loss = fluid.layers.cross_entropy(input=prediction, label=label)
    avg_loss = fluid.layers.mean(loss)
    return avg_loss

before_prog = fluid.Program()
startup = fluid.Program()

with fluid.program_guard(before_prog, startup):
    loss = resnet50_loss()
    opt = fluid.optimizer.Adam(learning_rate=0.001)
    opt.minimize(loss)

before_graph = IrGraph(core.Graph(before_prog.desc), for_test=False)
after_prog = before_graph.to_program()

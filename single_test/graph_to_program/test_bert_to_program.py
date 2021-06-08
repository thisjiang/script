import paddle
import paddle.fluid as fluid
from paddle.fluid import core
from paddle.fluid.framework import IrGraph
import six
from paddlenlp.transformers import BertModel
from common import *

paddle.enable_static()

def bert_loss():
    data = fluid.layers.data(name='input_ids', shape=[512], dtype='int32')
    label = fluid.layers.data(name='label', shape=[1], dtype='int64')
    model, _ = BertModel.from_pretrained('bert-wwm-chinese')
    prediction, _ = model.forward(data)
    loss = fluid.layers.cross_entropy(input=prediction, label=label)
    avg_loss = fluid.layers.mean(loss)
    return avg_loss

before_prog = fluid.Program()
startup = fluid.Program()

with fluid.program_guard(before_prog, startup):
    loss = bert_loss()
    opt = fluid.optimizer.Adam(learning_rate=0.001)
    opt.minimize(loss)


before_graph = IrGraph(core.Graph(before_prog.desc), for_test=False)
after_prog = before_graph.to_program()

# print program information
print_program(before_prog, "bert_toprogram_single_before.txt")
print_program(after_prog, "bert_toprogram_single_after.txt")
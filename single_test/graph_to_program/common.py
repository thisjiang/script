from paddle.fluid import core

# write to file
FILE_PATH = "/Paddle/GraphToProgram/"

def print_program(program, filename:str) -> None:
    # print program information
    with open(FILE_PATH + filename, "w+") as file:
        file.write(program.to_string(False))

def print_parameters(program, filename:str) -> None:
    with open(FILE_PATH + filename, "w+") as file:
        for param in program.all_parameters():
            file.write(param.to_string(False))
            file.write("\n")

def print_vars(program, filename:str) -> None:
    with open(FILE_PATH + filename, "w+") as file:
        for var in program.list_vars():
            file.write(var.to_string(False))
            file.write("\n")


# insert ops into program
def insert_ops(program) -> None:
    OpRole = core.op_proto_and_checker_maker.OpRole

    OP_ROLE_KEY = core.op_proto_and_checker_maker.kOpRoleAttrName()
    OP_ROLE_VAR_KEY = core.op_proto_and_checker_maker.kOpRoleVarAttrName()

    def is_update_op(op):
        return 'Param' in op.input_names and 'Grad' in op.input_names and \
                "LearningRate" in op.input_names

    def is_loss_grad_op(op):
        if OP_ROLE_KEY not in op.attr_names:
            return False
        op_role = int(op.all_attrs()[OP_ROLE_KEY])
        return op_role & int(OpRole.Backward) and op_role & int(OpRole.Loss)

    def is_backward_op(op):
        return OP_ROLE_KEY in op.attr_names and \
                int(op.all_attrs()[OP_ROLE_KEY]) & int(OpRole.Backward)

    def is_optimizer_op(op):
        return OP_ROLE_KEY in op.attr_names and \
                int(op.all_attrs()[OP_ROLE_KEY]) & int(OpRole.Optimize)

    def __insert_ops(main_prog):
        block = main_prog.global_block()
        ring_id = 0
        grad = None
        for idx, op in reversed(list(enumerate(block.ops))):
            if is_backward_op(op) and OP_ROLE_VAR_KEY in op.attr_names:
                op_role_var = op.attr(OP_ROLE_VAR_KEY)
                if len(op_role_var) == 0:
                    continue
                assert len(op_role_var) % 2 == 0
                offset = 1
                for i in range(0, len(op_role_var), 2):
                    param_name = op_role_var[i]
                    param = block.var(param_name)
                    grad_name = op_role_var[i + 1]
                    grad = block.var(grad_name)

            if is_loss_grad_op(op):
                nranks = 4
                loss_grad_var = block.vars[op.output_arg_names[0]]
                block._insert_op(
                    idx + 1,
                    type='scale',
                    inputs={'X': loss_grad_var},
                    outputs={'Out': loss_grad_var},
                    attrs={
                        'scale': 1.0 / nranks,
                        OP_ROLE_KEY: OpRole.Backward
                    })

    __insert_ops(program)
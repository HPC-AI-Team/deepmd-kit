#!/usr/bin/env python3

from tensorflow.python.framework import ops
from deepmd.env import op_grads_module
from deepmd.env import tf



# y = tanh(x)
# dx = dy * (1 - tanh(x)^2) = dy * (1 - y^2)
@ops.RegisterGradient("FastTanh")
def _fast_tanh_grad(op, dy) :
    # return [dy * (1 - tf.square(tf.tanh(op.inputs[0])))]
    return [dy * (1 - tf.square(op.outputs[0]))]
    # return op_grads_module.fast_tanh_grad(op.outputs[0],dy)


# c  = b  * (1 - a^2)
# da = dc * (-2*a*b)
# db = dc * (1 - a^2)

# @ops.RegisterGradient("FastTanhGrad")
# def _fast_tanh_grad_grad(op, dc):
#     da = dc * (-2 * op.inputs[0] * op.inputs[1])
#     db = dc * (1 - tf.square(op.inputs[0]))
#     return [da,db]
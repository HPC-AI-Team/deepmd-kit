#!/usr/bin/env python3

from tensorflow.python.framework import ops
from deepmd.env import op_grads_module
from deepmd.env import op_module
from deepmd.env import tf

# xw + b = y
@ops.RegisterGradient("GemmLayer")
def _gemm_layer_grad (op, dy) :
    dx = op_module.matmul_nt(dy, op.inputs[1])
    dw = op_module.matmul_tn(op.inputs[0], dy)
    db = tf.reduce_sum(dy,axis=0)
    return [dx, dw, db]

# tanh(xw + b) = z
@ops.RegisterGradient("GemmTanhLayer")
def _gemm_layer_grad (op, dz) :
    dy = dz * (1 - tf.square(op.outputs[0]))
    dx = op_module.matmul_nt(dy, op.inputs[1])
    dw = op_module.matmul_tn(op.inputs[0], dy)
    db = tf.reduce_sum(dy,axis=0)
    return [dx, dw, db]

# C = AB
@ops.RegisterGradient("MatmulNn")
def _matmul_nn_grad(op, dc):
    da = op_module.matmul_nt(dc,op.inputs[1])
    db = op_module.matmul_tn(op.inputs[0],dc)
    return [da,db]

# C = A(B^T)
@ops.RegisterGradient("MatmulNt")
def _matmul_nt_grad(op, dc):
    # da = tf.matmul(dc,op.inputs[1])
    # db = tf.matmul(dc,op.inputs[0],transpose_a=True)
    da = op_module.matmul_nn(dc,op.inputs[1])
    db = op_module.matmul_tn(dc,op.inputs[0])
    return [da,db]

# C = (A^T)B
@ops.RegisterGradient("MatmulTn")
def _matmul_tn_grad(op, dc):
    da = op_module.matmul_nt(op.inputs[1],dc)
    db = op_module.matmul_nn(op.inputs[0],dc)
    return [da,db]

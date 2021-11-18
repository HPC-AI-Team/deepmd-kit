"""Module used for transfering parameters between models."""

from typing import Dict, Optional, Sequence, Tuple
from deepmd.env import tf
import re
import numpy as np
import logging
import sys
import os
import matplotlib.pyplot as plt

__all__ = ["transfer"]

log = logging.getLogger(__name__)

PRECISION_MAPPING: Dict[int, type] = {
    1: np.float32,
    2: np.float64,
    19: np.float16,
}

table_path="table"
draw_path="draw"
result_path="res"
diff_path="diff"
diff_float_path="diff/float"
diff_half_path="diff/half"
diff_half_one_path="diff/one_half"
diff_half_a0_path="diff/a0_half"
diff_half_a1_path="diff/a1_half"
diff_half_a2_path="diff/a2_half"
diff_half_a3_path="diff/a3_half"
diff_half_a4_path="diff/a4_half"
diff_half_a5_path="diff/a5_half"

table_info_dict=dict()
table_dict=dict()
last_layer_size = 128


def load_graph(graph_name: str) -> tf.Graph:
    """Load graph from passed in path.

    Parameters
    ----------
    graph_name : str
        path to frozen graph on disk

    Returns
    -------
    tf.Graph
        tf graph object
    """
    graph_def = tf.GraphDef()
    with open(graph_name, "rb") as f:
        graph_def.ParseFromString(f.read())
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name="")
        return graph

def record_table_info(node):
    tabulate_table_info_node_pattern = re.compile(r"filter_type_\d/TabulateFusion(_\d)?/table_info")
    if tabulate_table_info_node_pattern.fullmatch(node.name) is None:
        return
    print(f"record_table_info match node : {node.name}")
    print("recording tabulate table info")
    tensor_proto = node.attr["value"].tensor
    node_type = PRECISION_MAPPING[tensor_proto.dtype]
    file_name = node.name.replace("/","_")
    if node_type == np.float64:
        array = np.frombuffer(tensor_proto.tensor_content,dtype=np.float64)
        table_info_dict[node.name] = array
    elif node_type == np.float32:
        array = np.frombuffer(tensor_proto.tensor_content,dtype=np.float32)
        table_info_dict[node.name] = array
    else :
        print("Not support type !!!")
        assert False

def locate_xx(lower,upper,_max,stride0,stride1,xx) -> (float,int):
    if xx < lower:
        table_idx = 0
        xx = 0
    elif xx < upper:
        table_idx = int((xx - lower) / stride0)
        xx = xx - (table_idx * stride0 + lower)
    elif xx < _max:
        first_stride = int((upper - lower) / stride0)
        table_idx = first_stride + int((xx - upper) / stride1)
        xx = xx - ((table_idx - first_stride) * stride1 + upper)
    else :
        table_idx = int((upper - lower) / stride0) + int((_max - upper) / stride1) - 1
        xx = 0
    return (xx,table_idx)


def record_tabulate_table(node):
    tabulate_table_node_pattern = re.compile(r"filter_type_\d/TabulateFusion(_\d)?/table")
    if tabulate_table_node_pattern.fullmatch(node.name) is None:
        return
    print(f"record_tabulate_table match node : {node.name}")
    print("recording tabulate table")
    tensor_proto = node.attr["value"].tensor
    node_type = PRECISION_MAPPING[tensor_proto.dtype]
    file_name = node.name.replace("/","_")

    if node_type == np.float64:
        array = np.frombuffer(tensor_proto.tensor_content,dtype=np.float64)
        array = array.reshape([-1,last_layer_size,6])
        table_dict[node.name] = array
    elif node_type == np.float32:
        array = np.frombuffer(tensor_proto.tensor_content,dtype=np.float32)
        array = array.reshape([-1,last_layer_size,6])
        table_dict[node.name] = array
    else :
        print("Not support type !!!")
        assert False

def visualize_tables():
    for name,array in table_dict.items():
        file_name = name.replace("/","_")
        for k in range(last_layer_size):
            file_path = os.path.join(draw_path,f"{file_name}_{k}.png")
            array_ = array[:,k,:]
            x_data = np.arange(array_.shape[0])
            fig = plt.figure()
            plt.plot(x_data,array_[:,0],label="a0",figure=fig)
            plt.plot(x_data,array_[:,1],label="a1",figure=fig)
            plt.plot(x_data,array_[:,2],label="a2",figure=fig)
            plt.plot(x_data,array_[:,3],label="a3",figure=fig)
            plt.plot(x_data,array_[:,4],label="a4",figure=fig)
            plt.plot(x_data,array_[:,5],label="a5",figure=fig)
            plt.legend()
            plt.savefig(file_path)

def write_tables():
    for name,array in table_dict.items():
        file_name = name.replace("/","_")
        for k in range(last_layer_size):
            file_path = os.path.join(table_path,f"{file_name}_{k}.csv")
            array_ = array[:,k,:]
            print(f"writing table to {file_path}")
            with open(file_path,"w") as f:
                f.write(f"a0,a1,a2,a3,a4,a5\n")
                for r in range(array_.shape[0]):
                    f.write(f"{array_[r,0]},{array_[r,1]},{array_[r,2]},{array_[r,3]},{array_[r,4]},{array_[r,5]}\n")

def analyze_tables():
    for name,array in table_dict.items():
        table = array
        table_info = table_info_dict[name + "_info"]
        analyze_table(table,table_info)
        break

def analyze_table(table,table_info):
    print(table.shape)
    print(f"a0 max : {np.max(table[:,:,0])}")
    print(f"a1 max : {np.max(table[:,:,1])}")
    print(f"a2 max : {np.max(table[:,:,2])}")
    print(f"a3 max : {np.max(table[:,:,3])}")
    print(f"a4 max : {np.max(table[:,:,4])}")
    print(f"a5 max : {np.max(table[:,:,5])}")

    print(f"a0 mean : {np.mean(table[:,:,0])}")
    print(f"a1 mean : {np.mean(table[:,:,1])}")
    print(f"a2 mean : {np.mean(table[:,:,2])}")
    print(f"a3 mean : {np.mean(table[:,:,3])}")
    print(f"a4 mean : {np.mean(table[:,:,4])}")
    print(f"a5 mean : {np.mean(table[:,:,5])}")

    print(f"a0 min : {np.min(table[:,:,0])}")
    print(f"a1 min : {np.min(table[:,:,1])}")
    print(f"a2 min : {np.min(table[:,:,2])}")
    print(f"a3 min : {np.min(table[:,:,3])}")
    print(f"a4 min : {np.min(table[:,:,4])}")
    print(f"a5 min : {np.min(table[:,:,5])}")

    lower = table_info[0]
    upper = table_info[1]
    _max = table_info[2]
    stride0 = table_info[3]
    stride1 = table_info[4]
    # print(table_info)
    # print(table)
    input = np.linspace(-1,45,4600)
    xx_ = np.empty_like(input)
    for i in range(input.size):
        input_ = input[i]
        xx, table_indx = locate_xx(lower,upper,_max,stride0,stride1,input_)
        xx_[i] = xx
    xx2_ = xx_  * xx_
    xx3_ = xx2_ * xx_
    xx4_ = xx2_ * xx2_
    xx5_ = xx3_ * xx2_
    print(np.max(xx_))
    print(np.max(xx2_))
    print(np.max(xx3_))
    print(np.max(xx4_))
    print(np.max(xx5_))

def visualize_res():

    for name,table in table_dict.items():
        file_name = name.replace("/","_")
        table_info = table_info_dict[name + "_info"]
        lower = table_info[0]
        upper = table_info[1]
        _max = table_info[2]
        stride0 = table_info[3]
        stride1 = table_info[4]

        for k in range(last_layer_size):
            res_file_path = os.path.join(result_path,f"{file_name}_{k}.png")
            diff_file_path = os.path.join(diff_path,f"{file_name}_{k}.png")
            diff_float_file_path = os.path.join(diff_float_path,f"{file_name}_{k}.png")
            diff_half_file_path = os.path.join(diff_half_path,f"{file_name}_{k}.png")
            diff_half_one_file_path = os.path.join(diff_half_one_path,f"{file_name}_{k}.png")
            diff_half_a0_file_path = os.path.join(diff_half_a0_path,f"{file_name}_{k}.png")
            diff_half_a1_file_path = os.path.join(diff_half_a1_path,f"{file_name}_{k}.png")
            diff_half_a2_file_path = os.path.join(diff_half_a2_path,f"{file_name}_{k}.png")
            diff_half_a3_file_path = os.path.join(diff_half_a3_path,f"{file_name}_{k}.png")
            diff_half_a4_file_path = os.path.join(diff_half_a4_path,f"{file_name}_{k}.png")
            diff_half_a5_file_path = os.path.join(diff_half_a5_path,f"{file_name}_{k}.png")

            print(f"writing {file_name}")

            input = np.linspace(-1,45,46000)
            xx_double = np.empty_like(input)
            a0_double = np.empty_like(input)
            a1_double = np.empty_like(input)
            a2_double = np.empty_like(input)
            a3_double = np.empty_like(input)
            a4_double = np.empty_like(input)
            a5_double = np.empty_like(input)
            for i in range(input.size):
                input_ = input[i]
                xxx, table_indx = locate_xx(lower,upper,_max,stride0,stride1,input_)
                xx_double[i] = xxx
                a0_double[i] = table[table_indx, k, 0]
                a1_double[i] = table[table_indx, k, 1]
                a2_double[i] = table[table_indx, k, 2]
                a3_double[i] = table[table_indx, k, 3]
                a4_double[i] = table[table_indx, k, 4]
                a5_double[i] = table[table_indx, k, 5]
            res_double = a0_double + (a1_double + (a2_double + (a3_double + (a4_double + a5_double * xx_double) * xx_double) * xx_double) * xx_double) * xx_double

            xx_float = xx_double.astype(np.float32)
            a0_float = a0_double.astype(np.float32)
            a1_float = a1_double.astype(np.float32)
            a2_float = a2_double.astype(np.float32)
            a3_float = a3_double.astype(np.float32)
            a4_float = a4_double.astype(np.float32)
            a5_float = a5_double.astype(np.float32)
            res_float = a0_float + (a1_float + (a2_float + (a3_float + (a4_float + a5_float * xx_float) * xx_float) * xx_float) * xx_float) * xx_float

            xx_half = xx_double.astype(np.float16)
            a0_half = a0_double.astype(np.float16)
            a1_half = a1_double.astype(np.float16)
            a2_half = a2_double.astype(np.float16)
            a3_half = a3_double.astype(np.float16)
            a4_half = a4_double.astype(np.float16)
            a5_half = a5_double.astype(np.float16)
            res_half = a0_half + (a1_half + (a2_half + (a3_half + (a4_half + a5_half * xx_half) * xx_half) * xx_half) * xx_half) * xx_half

            # res
            fig = plt.figure()
            plt.plot(input,res_double,label="double",figure=fig)
            plt.plot(input,res_float,label="float",figure=fig)
            plt.plot(input,res_half,label="half",figure=fig)
            plt.legend()
            plt.savefig(res_file_path)
            
            # diff half float
            diff_float = res_double - res_float.astype(np.float64)
            diff_half = res_double - res_half.astype(np.float64)

            fig = plt.figure()
            plt.plot(input,diff_float,label="float",figure=fig)
            plt.plot(input,diff_half,label="half",figure=fig)
            plt.legend()
            plt.savefig(diff_file_path)
            fig = plt.figure()
            plt.plot(input,diff_float,label="float",figure=fig)
            plt.legend()
            plt.savefig(diff_float_file_path)
            fig = plt.figure()
            plt.plot(input,diff_half,label="half",figure=fig)
            plt.legend()
            plt.savefig(diff_half_file_path)

            # diff a0 a1 a2 a3 a4 a5 half
            # res_double  = a0_double + (a1_double + (a2_double + (a3_double + (a4_double + a5_double * xx_double) * xx_double) * xx_double) * xx_double) * xx_double
            res_half_a0 = a0_half + (a1_double + (a2_double + (a3_double + (a4_double + a5_double * xx_double) * xx_double) * xx_double) * xx_double) * xx_double
            res_half_a1 = a0_double + (a1_half + (a2_double + (a3_double + (a4_double + a5_double * xx_double) * xx_double) * xx_double) * xx_double) * xx_double
            res_half_a2 = a0_double + (a1_double + (a2_half + (a3_double + (a4_double + a5_double * xx_double) * xx_double) * xx_double) * xx_double) * xx_double
            res_half_a3 = a0_double + (a1_double + (a2_double + (a3_half + (a4_double + a5_double * xx_double) * xx_double) * xx_double) * xx_double) * xx_double
            res_half_a4 = a0_double + (a1_double + (a2_double + (a3_double + (a4_half + a5_double * xx_double) * xx_double) * xx_double) * xx_double) * xx_double
            res_half_a5 = a0_double + (a1_double + (a2_double + (a3_double + (a4_double + a5_half * xx_double) * xx_double) * xx_double) * xx_double) * xx_double

            diff_half_a0 = res_double - res_half_a0.astype(np.float64)
            diff_half_a1 = res_double - res_half_a1.astype(np.float64)
            diff_half_a2 = res_double - res_half_a2.astype(np.float64)
            diff_half_a3 = res_double - res_half_a3.astype(np.float64)
            diff_half_a4 = res_double - res_half_a4.astype(np.float64)
            diff_half_a5 = res_double - res_half_a5.astype(np.float64)
            
            #diff a0
            fig = plt.figure()
            plt.plot(input,diff_half_a0,label="a0",figure=fig)
            plt.legend()
            plt.savefig(diff_half_a0_file_path)   

            #diff a1
            fig = plt.figure()
            plt.plot(input,diff_half_a1,label="a1",figure=fig)
            plt.legend()
            plt.savefig(diff_half_a1_file_path)   

            #diff a2
            fig = plt.figure()
            plt.plot(input,diff_half_a2,label="a2",figure=fig)
            plt.legend()
            plt.savefig(diff_half_a2_file_path)   

            #diff a3
            fig = plt.figure()
            plt.plot(input,diff_half_a3,label="a3",figure=fig)
            plt.legend()
            plt.savefig(diff_half_a3_file_path)   

            #diff a4
            fig = plt.figure()
            plt.plot(input,diff_half_a4,label="a4",figure=fig)
            plt.legend()
            plt.savefig(diff_half_a4_file_path)   

            #diff a5
            fig = plt.figure()
            plt.plot(input,diff_half_a5,label="a5",figure=fig)
            plt.legend()
            plt.savefig(diff_half_a5_file_path)   


def analyze(model: str):
    graph = load_graph(model)
    print(f"{len(graph.as_graph_def().node)} ops in the graph")
    graph_def = graph.as_graph_def()

    for node in graph_def.node:
        record_table_info(node)
        record_tabulate_table(node)
    
    # visualize_tables()
    # write_tables()
    visualize_res()

if __name__ == "__main__" :
    if len(sys.argv) != 3:
        print(f"{sys.argv[0]} model output")
    model_path = sys.argv[1]
    print(f"model path : {model_path}")
    analyze(model_path)
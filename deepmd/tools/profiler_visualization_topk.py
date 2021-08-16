from deepmd.env import tf
from tensorflow.python.platform import gfile
from graphviz import Digraph
import sys
import json
import os

def time_dict_from_profile_dict(profile_dict):
    time_dict = dict()
    for op in profile_dict["traceEvents"][1:]:
        name = op["name"]
        dur = op["dur"]
        time_dict[name] = dur
    return time_dict


def time_list_from_multi_profile_file(profile_name):
    N = 0
    while True:
        profile_path = "{}.json_{}".format(profile_name, N)
        if os.path.exists(profile_path):
            N += 1
        else:
            break
    time_dict = {}
    for i in range(N):
        profile_path = "{}.json_{}".format(profile_name, i)
        with open(profile_path,'r') as f:
            time_dict_tmp = time_dict_from_profile_dict(json.loads(f.read()))
            for k,v in time_dict_tmp.items():
                if k in time_dict:
                    time_dict[k].append(v)       
                else:
                    time_dict[k] = [v]
    time_list = []
    for k,vs in time_dict.items():
        total = 0
        min_value = 10000000
        max_value = 0
        for v in vs:
            total += v
            min_value = min(min_value,v)
            max_value = max(max_value,v)
        assert(len(vs) == N)
        agv = total/N
        time_list.append((k, agv, min_value, max_value))
    time_list.sort(key= lambda x:x[1],reverse=True)
    return time_list


def compute_total_time(time_list):
    total_time = 0
    for name,agv,_,_ in time_list:
        total_time += agv
    return total_time


def print_topk_ops(time_list,total_time,k = 100):
    print()
    print("Top k op")
    print("{:8}\t{:8}\t{:8}\t{:8}\t{}".format("average time","min time","max time","percentage","op name"))
    for i in range(min(k,len(time_list))):
        print("{:8.2f}\t{:8}\t{:8}\t{:8.4f}\t{}".format(time_list[i][1],time_list[i][2],time_list[i][3],time_list[i][1]/total_time,time_list[i][0]))
        
def get_op_type_from_name(name):
    return name.split('/')[-1].split('_')[0]

def group_by_op_type(time_list):
    time_dict = dict()
    for name, agv, min_time, max_time in time_list:
        op_type = get_op_type_from_name(name)
        if op_type in time_dict:
            time_dict[op_type] = time_dict[op_type] + agv
        else:
            time_dict[op_type] = agv
    return time_dict


def time_list_for_op_type(time_list):
    time_dict = group_by_op_type(time_list)
    time_list = list()
    for k,v in time_dict.items():
        time_list.append((k,v));     
    time_list.sort(key= lambda x:x[1],reverse=True)
    return time_list


def print_topk_op_type(op_type_time_list,total_time,k = 20):
    print()
    print("Top k op type")
    print("{:8}\t{:8}\t{}".format("average time", "percentage", "op type"))
    for i in range(min(k,len(op_type_time_list))):
        print("{:8.2f}\t{:8.4f}\t{}".format(op_type_time_list[i][1],op_type_time_list[i][1]/total_time,op_type_time_list[i][0]))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("{} profile_name".format(sys.argv[0]))
        exit(-1)
    profile_name = sys.argv[1]
    time_list = time_list_from_multi_profile_file(profile_name)
    total_time = compute_total_time(time_list)
    print_topk_ops(time_list,total_time)

    op_type_time_list = time_list_for_op_type(time_list)
    print_topk_op_type(op_type_time_list,total_time)
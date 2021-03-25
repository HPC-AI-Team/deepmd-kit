
from deepmd.DeepPot import DeepPot
import numpy as np

def init_model(model_path):
    # print("in init model")
    # print(model_path)
    return DeepPot(model_path)

def get_scalar(model,name):
    # print("in get scalar !!!")
    # print(name)
    # TODO
    name = "load/"+name+":0"
    ret = model.sess.run(model.graph.get_tensor_by_name(name))
    return ret

def infer(model,t_coord,t_type,t_box,t_mesh,t_natoms):
    return model.infer(t_coord,t_type,t_box,t_mesh,t_natoms)

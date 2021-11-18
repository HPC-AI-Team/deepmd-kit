from deepmd.env import tf
from google.protobuf import text_format
from tensorflow.python.platform import gfile
import sys


def pb2pbtxt(*,input_model: str,output_model: str,**kwargs):
    graphdef_to_pbtxt(input_model,output_model)

def graphdef_to_pbtxt(pb_path,pbtxt_path):
    with gfile.FastGFile(pb_path,'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def,name="")
        tf.train.write_graph(graph_def,"./",pbtxt_path,as_text=True)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("{} pb_path pbtxt_path".format(sys.argv[0]))
    pb_path = sys.argv[1]
    pbtxt_path = sys.argv[2]
    graphdef_to_pbtxt(pb_path,pbtxt_path)
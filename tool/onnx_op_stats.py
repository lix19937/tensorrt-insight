# /**************************************************************
#  * @Author: lix19937 
#  * @Date: 2021-09-15 10:03:26 
#  * @Last Modified by:   lix19937 
#  * @Last Modified time: 2021-09-15 10:03:26 
#  **************************************************************/

import onnx
import prettytable as pt

def dump_onnx(onnx_file):
    onnx_model = onnx.load(onnx_file) ##     
    graph = onnx_model.graph
    nodes = graph.node

    op_num_map = {}

    tb = pt.PrettyTable()
    tb.field_names = ["op name", "num"]
  
    for node in nodes:
        if node.op_type not in op_num_map:
            op_num_map[node.op_type] = 1
        else:
            op_num_map[node.op_type] += 1 

    sorted_by_key = dict(sorted(op_num_map.items()))

    for k,v in sorted_by_key.items():
        tb.add_row([k, v])

    print(tb)

dump_onnx("refnet_nuplan_mini_250116.onnx")


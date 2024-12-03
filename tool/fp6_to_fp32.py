'''
onnx==1.8.1
numpy==1.18.5
typer==0.3.2
'''

# python3 convert.py fp16_model.onnx ./converted_fp32_model.onnx


import onnx

from onnx import helper as h
from onnx import checker as ch
from onnx import TensorProto, GraphProto
from onnx import numpy_helper as nph

import numpy as np
from collections import OrderedDict

from logger import log
import typer

def make_param_dictionary(initializer):
    params = OrderedDict()
    for data in initializer:
        params[data.name] = data
    return params

def convert_params_to_fp32(params_dict):
    converted_params = []
    for param in params_dict:
        data = params_dict[param]
        if data.data_type == TensorProto.FLOAT16:
            data_cvt = nph.to_array(data).astype(np.float32)
            data = nph.from_array(data_cvt, data.name)
        converted_params += [data]
    return converted_params

def convert_constant_nodes_to_fp32(nodes):
    """
    convert_constant_nodes_to_fp32 Convert Constant nodes to FLOAT32. If a constant node has data type FLOAT16, a new version of the
    node is created with FLOAT32 data type and stored.

    Args:
      nodes (list): list of nodes

    Returns:
      list: list of new nodes all with FLOAT32 constants.
    """
    new_nodes = []
    for node in nodes:
        if (
            node.op_type == "Constant"
            and node.attribute[0].t.data_type == TensorProto.FLOAT16
        ):
            data = nph.to_array(node.attribute[0].t).astype(np.float32)
            new_t = nph.from_array(data)
            new_node = h.make_node(
                "Constant",
                inputs=[],
                outputs=node.output,
                name=node.name,
                value=new_t,
            )
            new_nodes += [new_node]
        else:
            new_nodes += [node]

    return new_nodes

def convert_model_to_fp32(model_path: str, out_path: str):
    """
    convert_model_to_fp32 Converts ONNX model with FLOAT16 params to FLOAT32 params.\n

    Args:\n
      model_path (str): path to original ONNX model.\n
      out_path (str): path to save converted model.
    """
    log.info("ONNX FLOAT16 --> FLOAT32 Converter")
    log.info(f"Loading Model: {model_path}")
    # * load model.
    model = onnx.load_model(model_path)
    ch.check_model(model)
    # * get model opset version.
    opset_version = model.opset_import[0].version
    graph = model.graph
    # * convert all FLOAT16 input/output to FLOAT32.
    for input in model.graph.input:
        input.type.tensor_type.elem_type = 1
    for output in model.graph.output:
        output.type.tensor_type.elem_type = 1
    # * The initializer holds all non-constant weights.
    init = graph.initializer
    # * collect model params in a dictionary.
    params_dict = make_param_dictionary(init)
    log.info("Converting FLOAT16 model params to FLOAT32...")
    # * convert all FLOAT16 aprams to FLOAT32.
    converted_params = convert_params_to_fp32(params_dict)
    log.info("Converting constant FLOAT16 nodes to FLOAT32...")
    new_nodes = convert_constant_nodes_to_fp32(graph.node)

    graph_name = f"{graph.name}-fp32"
    log.info("Creating new graph...")
    # * create a new graph with converted params and new nodes.
    graph_fp32 = h.make_graph(
        new_nodes,
        graph_name,
        graph.input,
        graph.output,
        initializer=converted_params,
    )
    log.info("Creating new float32 model...")
    model_fp32 = h.make_model(graph_fp32, producer_name="onnx-typecast")
    model_fp32.opset_import[0].version = opset_version
    ch.check_model(model_fp32)
    log.info(f"Saving converted model as: {out_path}")
    onnx.save_model(model_fp32, out_path)
    log.info(f"Done Done London. 🎉")
    return

if __name__ == "__main__":
    typer.run(convert_model_to_fp32)

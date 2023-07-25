#!/usr/bin/env python3
#

import onnx_graphsurgeon as gs
import numpy as np
import onnx
import argparse
from struct import pack, unpack
from typing import Dict, Union, Set, List
from loguru import logger
import collections

# DOS6030 use TRT8410! If you work in your own TensorRT, Change it to suit your version!!
TRT_VERSION = "8410" 

def load_calib(calib_file):
  logger.info("calib_file:{}".format(calib_file))
  f = open(calib_file, "r")
  logger.info("{}".format(type(f)))
  line = f.readline().rstrip()

  calib_map = collections.OrderedDict()
  while True:
    line = f.readline().rstrip()
    if not line:
      break;
    vec = line.split(":")    
    assert len(vec) == 2, "error line %s" % line  
    calib_map[vec[0]] = vec[1] 
  f.close()
  return calib_map

def merge_calib(qat_calib, ptq_calib):   
  qat_map = load_calib(qat_calib)
  ptq_map = load_calib(ptq_calib)
  logger.info("CalibrationTable QAT num_layer:{}".format(len(qat_map)) )
  logger.info("CalibrationTable PTQ num_layer:{}".format(len(ptq_map)) )

  merge_map = collections.OrderedDict()
  diff_list = []

  for k,v in ptq_map.items():
    if k in qat_map:
      merge_map[k] = qat_map[k]  
    else:
      merge_map[k] = v
      diff_list.append(k)

  fCalibrationTable = open(qat_calib + "_fused_ptq", 'w')
  fCalibrationTable.write("TRT-"+str(TRT_VERSION)+"-EntropyCalibration2\n")

  num_layer = 0
  for k,v in merge_map.items():
      fCalibrationTable.write(k+': ' + v + '\n')
      num_layer = num_layer + 1    
  fCalibrationTable.close()
  logger.info("CalibrationTable merge num_layer:{}".format(num_layer) )


def get_quantized_tensor(node: gs.Node, graph: gs.Graph) -> Union[gs.Variable, gs.Constant]:
        def convert_constant_to_variable_node(nodes_to_convert: List[gs.Node]):
            for node_input in nodes_to_convert:
                # Copy Constant into temporary variable
                node_input_copy = gs.Constant(name=node_input.name + "_constant", values=node_input.values,
                                              data_location=node_input.data_location)
                # Make Constant Node and append to 'graph'
                node_input_copy_node = gs.Node(op="Constant", attrs={'value': node_input_copy},
                                               inputs=[], outputs=[node_input_copy])
                graph.nodes.append(node_input_copy_node)
                # Convert original Constant to Variable type with the copied Constant as input
                node_input.to_variable(dtype=node_input.dtype, shape=node_input.shape)
                node_input.inputs.append(node_input_copy_node)

        if not node.op == "QuantizeLinear" or len(node.inputs) != 3:
            raise RuntimeError(f"Expected QuantizeLinear with 3 arguments, but got {node.op} with "
                               f"{len(node.inputs)} arguments.")
        # For weight quantizers: Exported as per-channel QuantLinear operators, `x` and
        # `y_zero_point` are parsed as gs.Constants and `y_scale` is a gs.Variable filled by a
        # gs.Constant operator.
        if type(node.inputs[0]) == gs.Constant:
            if type(node.inputs[1]) == gs.Constant:
                convert_constant_to_variable_node([node.inputs[1]])
            quantize_tensor = node.inputs[0]
        # For activation quantizers: Exported as per-tensor QuantizeLinear operators, `x`, `y_scale`
        # and `y_zero_point` are all parsed to gs.Variables and scale and zero-point are filled by
        # gs.Constant operators.
        else:
            nodes_to_convert = [node_input for node_input in node.inputs if type(node_input) == gs.Constant]
            convert_constant_to_variable_node(nodes_to_convert)

            quantize_tensor = node.inputs[0]
        return quantize_tensor

#
# python3 QAT2PTQ.py --model qat1112_3head.onnx  
#
if __name__ == "__main__":
    # merge_calib(qat_calib="CalibrationTableqat1112_3head", ptq_calib="qat1112_3headmodified.onnx.calib")
    # exit(0)
    parser = argparse.ArgumentParser(description='Small tool for convert QAT model to pf32 model with CalibrationTable')
    parser.add_argument('--model', default='qat.onnx', type=str,help='the QAT model')
    args = parser.parse_args()
    graph = gs.import_onnx(onnx.load(args.model))

    # remove Q&DQ nodes
    Qnodes = [node for node in graph.nodes if node.op == 'QuantizeLinear']
    for OneNode in graph.nodes:
        for i, oneInputTensor in enumerate(OneNode.inputs):
            if len(oneInputTensor.inputs) != 1:
                continue
            if oneInputTensor.inputs[0].op != "DequantizeLinear":
                continue
            Qnode = oneInputTensor.inputs[0].i()

            Qnode_input = Qnode.inputs[0]
            OneNode.inputs[i] = Qnode_input

    # write calibrationTable to file
    CalibrationTablePath = str("CalibrationTable"+args.model).replace(".onnx",'')
    DynamicRangeTablePath = str("DynamicRange"+args.model).replace(".onnx",'')
    fCalibrationTable = open(CalibrationTablePath, 'w')
    fDynamicRange = open(DynamicRangeTablePath, 'w')
    
    num_layer = 0
    fCalibrationTable.write("TRT-"+str(TRT_VERSION)+"-EntropyCalibration2\n")
    # extract scale value
    for OneQ in Qnodes:
        quant_tensor = get_quantized_tensor(OneQ, graph)
        # judge activation-quanter
        is_activation_quantizer = len(quant_tensor.inputs) > 0
        is_input_quantizer = len(quant_tensor.inputs) == 0 and quant_tensor.name in [
            i.name for i in graph.inputs]
        
        # logger.info(is_input_quantizer)
        if is_activation_quantizer or is_input_quantizer:
            # This assumes the quantization for activation is per-tensor quantization.
            Scale = OneQ.inputs[1].inputs[0].attrs["value"].values
            dynamicRange = np.float32(127.0 * Scale)
            fDynamicRange.write(quant_tensor.name+ ': '+str(dynamicRange)+'\n')
            b = pack('f', Scale)
            fCalibrationTable.write(quant_tensor.name+': ' + str(hex(unpack('i', b)[0]))+'\n')
            num_layer = num_layer + 1
    
    fCalibrationTable.close()
    fDynamicRange.close()

    graph.cleanup().toposort()
    save_model_path = str(args.model).replace(".onnx",'') + "modified.onnx"
    logger.info("convert Done...\n" )
    logger.info("onnx model saved as:{}".format(save_model_path) )
    logger.info("CalibrationTable saved as:{}".format(CalibrationTablePath) )
    logger.info("CalibrationTable num_layer:{}".format(num_layer) )

    onnx.save(gs.export_onnx(graph), save_model_path)
    

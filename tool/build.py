
#  TensorRT-release-8.6/TensorRT-release-8.6/demo/DeBERTa/deberta_tensorrt_inference.py 
import os
import tensorrt as trt
import json
from loguru import logger as MyLOG


def build_engine(onnx_file_path, engine_file_path, layerinfo_json):
    trt_logger = trt.Logger(trt.Logger.VERBOSE)  
    builder = trt.Builder(trt_logger)
    network = builder.create_network(1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    
    parser = trt.OnnxParser(network, trt_logger)
    with open(onnx_file_path, 'rb') as model:
        if not parser.parse(model.read()):
            print('ERROR: Failed to parse the ONNX file.')
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None
    print("Completed parsing ONNX file")

    config = builder.create_builder_config()
    builder.max_batch_size = 1 
    print('num_layers:', network.num_layers) 

    if os.path.isfile(engine_file_path):
        try:
            os.remove(engine_file_path)
        except Exception:
            print("Cannot remove existing file: ",
                engine_file_path)

    print("Creating Tensorrt Engine") 

    # config.set_flag(trt.BuilderFlag.STRICT_TYPES)
    config.max_workspace_size = 2 << 30
    config.set_flag(trt.BuilderFlag.FP16)
    config.set_flag(trt.BuilderFlag.INT8)

    config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED

    engine = builder.build_engine(network, config)
    print("engine.__len__() = %d" % len(engine))
    print("engine.__sizeof__() = %d" % engine.__sizeof__())
    print("engine.__str__() = %s" % engine.__str__())

    with open(engine_file_path, "wb") as f:
        f.write(engine.serialize())
    print("Serialized Engine Saved at: ", engine_file_path)
    
    # runtime = trt.Runtime(trt_logger)
    # engine = runtime.deserialize_cuda_engine(serialized_engine)    
    
    inspector = engine.create_engine_inspector()
    layer_json = json.loads(inspector.get_engine_information(trt.LayerInformationFormat.JSON))
    with open(layerinfo_json, "w") as fj:
      json.dump(layer_json, fj)
    return engine


if __name__ == "__main__":
    logger_ = trt.Logger(trt.Logger.VERBOSE)
    trt.init_libnvinfer_plugins(logger_, "")

    MyLOG.info("{}".format(trt.__version__, trt.__file__))
    ONNX_SIM_MODEL_PATH = './yolox.onnx'

    TENSORRT_ENGINE_PATH = ONNX_SIM_MODEL_PATH + ".plan"

    eg = build_engine(ONNX_SIM_MODEL_PATH, TENSORRT_ENGINE_PATH, "./yolox_layer.json")
    MyLOG.info("Serialized Engine Done")


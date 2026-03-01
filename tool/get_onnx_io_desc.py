import onnx
from onnx import TensorProto

def get_onnx_io_info(onnx_model_path):
    try:
        model = onnx.load(onnx_model_path)
        
        input_infos = []
        for input_tensor in model.graph.input:
            name = input_tensor.name
            
            shape = []
            for dim in input_tensor.type.tensor_type.shape.dim:
                if dim.dim_value != 0:
                    shape.append(dim.dim_value)
                else:
                    shape.append(None)
            
            # 获取张量数据类型（转换为可读的字符串）
            dtype = TensorProto.DataType.Name(input_tensor.type.tensor_type.elem_type)
            
            input_infos.append({
                "name": name,
                "shape": shape,
                "type": dtype
            })
        
        output_infos = []
        for output_tensor in model.graph.output:
            name = output_tensor.name            
            shape = []
            for dim in output_tensor.type.tensor_type.shape.dim:
                if dim.dim_value != 0:
                    shape.append(dim.dim_value)
                else:
                    shape.append(None)
            
            dtype = TensorProto.DataType.Name(output_tensor.type.tensor_type.elem_type)            
            output_infos.append({
                "name": name,
                "shape": shape,
                "type": dtype
            })
        
        return {
            "inputs": input_infos,
            "outputs": output_infos
        }
    
    except FileNotFoundError:
        print(f"err, not found {onnx_model_path}")
        return None
    except Exception as e:
        print(f"load onnx err, {str(e)}")
        return None

if __name__ == "__main__":
    model_path = "./onnx_0915/onnx_1124/detr.onnx"

    io_info = get_onnx_io_info(model_path)
    if io_info:
        print("=" * 50)
        print("IN")
        print("=" * 50)
        for idx, input_info in enumerate(io_info["inputs"]):
            # print(f"输入 {idx+1}:")
            print(f"  名称: {input_info['name']}")
            print(f"  形状: {input_info['shape']}")
            print(f"  类型: {input_info['type']}")
            print()
        
        print("=" * 50)
        print("OUT")
        print("=" * 50)
        for idx, output_info in enumerate(io_info["outputs"]):
            # print(f"输出 {idx+1}:")
            print(f"  名称: {output_info['name']}")
            print(f"  形状: {output_info['shape']}")
            print(f"  类型: {output_info['type']}")
            print()

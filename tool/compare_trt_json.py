
import json
import numpy as np
import re

np.set_printoptions(precision=7, suppress=True)

# https://www.geeksforgeeks.org/convert-python-list-to-numpy-arrays/

def read_json_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            file_content = f.read()
            
            # nan/NaN/NAN ---> JSON null
            file_content = re.sub(r'\bnan\b', 'null', file_content, flags=re.IGNORECASE)
            # inf/-inf ----> 1-e999
            file_content = re.sub(r'\binf\b', '1e999', file_content, flags=re.IGNORECASE)
            file_content = re.sub(r'\b-inf\b', '-1e999', file_content, flags=re.IGNORECASE)

            def parse_nan_constant(constant):
                if constant.lower() == 'null':
                    return float('nan')
                elif constant == '1e999':
                    return float('inf')
                elif constant == '-1e999':
                    return float('-inf')
                return json.JSONDecoder().constants[constant]

            data = json.loads(file_content, parse_constant=parse_nan_constant)
        return data
    except FileNotFoundError:
        print(f"not found {file_path}")
        return None
    except json.JSONDecodeError as e:
        print(f"\nJSON parse error:")
        print(f"err type:{e.msg}")
        print(f"err loc: {e.lineno} row, {e.colno} column")
        return None

def custom_print(content, width=50, align='<', fill_char=' '):
    """
    :param content:  
    :param width:  
    :param align: < left; >right; ^mid
    :param fill_char:  
    """
    content_str = str(content)
    if len(content_str) > width:
        content_str = content_str[:width-3] + "..."
    formatted = f"{content_str:{fill_char}{align}{width}}"
    print(formatted, end = '')

   
def compare(json_list):
    print(json_list)
    dict_list = []

    for idx in range(len(json_list)):
        f = json_list[idx]
        print(">> ", f, end = ' ') 
        dict_list.append(read_json_file(f))
        print("done")
        
    dict_vec1 = dict_list[0]
    dict_vec2 = dict_list[1]

    for idx in range(len(dict_vec2)): 
        it1 = dict_vec1[idx]
        it2 = dict_vec2[idx]

        out1 = it1["values"]
        out2 = it2["values"]

        arr1 = np.asarray(out1)
        arr2 = np.asarray(out2)
        name = it1["name"]

        # print(name, type(arr1), type(arr1[0]), end=' ')        
        
        custom_print(name, width=32, align='<', fill_char=' ')
        custom_print(type(arr1), width=32, align='<', fill_char=' ')
        custom_print(type(arr1[0]), width=32, align='<', fill_char=' ')

        if (arr1 == None).any() or (arr2 == None).any():
            c = 0
        else:    
            c = arr1 - arr2
            
        # print(f"max:{np.max(c)}, min:{np.min(c)}")
        custom_print(np.max(c), width=24, align='<', fill_char=' ')
        custom_print(np.min(c), width=24, align='<', fill_char=' ')
        custom_print("\n", width=1, align='<', fill_char=' ')


json_list = ["./msda_torch_simp.json", "./msda_trt_self.json"]
compare(json_list)

# json_list = ["./msda_torch_simp.json", "./msda_trt_lhm.json"]
# compare(json_list)

# json_list = ["./msda_trt_lhm.json", "./msda_trt_self.json"]
# compare(json_list)

# json_list = ["./thoru_fp16_d_mt_model_1215_merge_simplify_plugin_0112_v1_out.json", 
#              "./thoru_fp16_d_mt_model_1215_merge_simplify_plugin_0112_v1_trt_out.json"]


compare(json_list)

print("done")

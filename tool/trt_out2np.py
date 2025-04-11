

import numpy as np
import json
import os 

# tensorrt output by --dumpOutput 
directory_path = "output_tensor/"
file_names = os.listdir(directory_path)

icnt = 0
for file in file_names:
  with open(directory_path + file, "r") as f:
      output_data = json.load(f)

  out_dir = f"output_tensor_npy/{icnt}/"
  if not os.path.exists(out_dir):
      os.makedirs(out_dir)

  idx = 0
  for it in output_data:
    print(f" {idx} {it.keys()}  {it['name']}  {it['dimensions']}  {type(it['values'])}" )
    np_array = np.array(it["values"])  
    np.save(f"{out_dir}{it['name']}.npy", np_array)
    idx+=1

  icnt+=1

print("done")

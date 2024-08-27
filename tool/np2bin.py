
# save numpy to bin for `trt --loadInputs=` use  

import numpy as np

data = np.asarray(img_data, dtype=np.float32)
data.tofile("sod_out/in.bin")

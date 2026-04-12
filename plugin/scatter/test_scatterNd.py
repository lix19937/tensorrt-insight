import torch 
import torch.nn as nn
import numpy as np

# https://blog.csdn.net/u011622208/article/details/115013246

use_mini_data = False

p=40000
nchannels=64
nx=608
ny=608
'''
canvas torch.Size([64, 369664])
indices torch.Size([40000])
features torch.Size([64, 40000])
'''
if use_mini_data:
    p=8
    nchannels=4
    nx=3
    ny=3
    
def generate_unique_points(n=p, width=nx, height=ny):
    if n > width * height:
        raise ValueError("overflow !")
    total_pixels = width * height
    
    # 2. from [0, total_pixels) get N diff value
    indices = np.random.choice(total_pixels, size=n, replace=False)
    
    # 3. 1d-2d
    x = indices % width
    y = indices // width
    
    # 4. pair to (N, 2) 
    # points = np.stack((x, y), axis=-1)    
    # return points
    return y, x

# -------------------------------------------------------------------
if use_mini_data:
    features = torch.zeros(nchannels, p,  dtype=torch.float32) + torch.arange(1, nchannels*p+1).view(nchannels, p)
else:
    features = torch.randn(nchannels, p,  dtype=torch.float32)

coors = torch.zeros(p, 4,  dtype=torch.int32)
y, x = generate_unique_points(p, nx, ny)

coors[:, 2] = torch.from_numpy(y)
coors[:, 3] = torch.from_numpy(x)

voxel_num = torch.tensor([1], dtype=torch.int32)

# -------------------------------------------------------------------
data = features.numpy();                    data.tofile("features.bin")
data = features.numpy().astype(np.float16); data.tofile("features_fp16.bin")

data = coors.numpy();                       data.tofile("coors.bin")
data = voxel_num.numpy();                   data.tofile("voxel_num.bin")        
# onnxsim ./model.onnx  model_sim.onnx

def scatter_nd(canvas, indices, updates):
    output = np.copy(canvas)
    update_indices = indices.shape[:-1]
    for idx in np.ndindex(update_indices):
        # output[tuple(indices[idx])] = updates[idx] !!!
        output[:, indices[idx]] = updates[idx]

    output_ref = np.copy(canvas)
    for idx in range(len(indices)):
        output_ref[:, indices[idx]] = updates[:, idx]
        
    return output


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, features, coors):
        canvas = torch.zeros(nchannels, nx * ny, dtype=torch.float32)

        indices = coors[:, 2] * nx + coors[:, 3]
        indices = indices.type(torch.long)
        canvas = torch.zeros(nchannels, nx*ny, dtype=torch.float32)

        # exp
        exp = scatter_nd(canvas.numpy(), indices.numpy(), features.numpy())

        # gt
        canvas[:, indices] = features

        print(torch.equal(torch.from_numpy(exp).view(-1, 1), canvas.view(-1, 1) ))

        exp.tofile("gt_out.bin")        

        batch_canvas = canvas.view(1, nchannels, ny, nx)
        return batch_canvas


model = SimpleModel().eval()

model(features, coors)

onnx_path = "model.onnx"

# torch.onnx.export(
#     model,
#     (features, coors),
#     onnx_path,
#     opset_version=17,  
#     input_names=["features", "coors"],
#     output_names=["output"],
# )

'''
coors[:, 2] = torch.tensor([0, 1, 2, 0, 1, 1, 2, 2], dtype=torch.int32)
coors[:, 3] = torch.tensor([1, 1, 0, 2, 0, 2, 2, 1], dtype=torch.int32)
	  0*3 +  1
      1*3 +  1
      2*3 +  0
      0*3 +  2
      1*3 +  0
      1*3 +  2
      2*3 +  2
      2*3 +  1  	
indices = torch.tensor([1, 4, 6, 2, 3, 5, 8, 7], dtype=torch.long)

indices                       P   total thread num <P, so  threadidx < P
tensor([1, 4, 6, 2, 3, 5, 8, 7])  

features                      C*P                  feature[:, threadidx]
tensor([[ 1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.],
        [ 9., 10., 11., 12., 13., 14., 15., 16.],
        [17., 18., 19., 20., 21., 22., 23., 24.],
        [25., 26., 27., 28., 29., 30., 31., 32.]])

canvas torch.Size([4, 9])     C, valid_num         canvas[:, indices[threadidx]]    
tensor([[ 0.,  1.,  4.,  5.,  2.,  6.,  3.,  8.,  7.],
        [ 0.,  9., 12., 13., 10., 14., 11., 16., 15.],
        [ 0., 17., 20., 21., 18., 22., 19., 24., 23.],
        [ 0., 25., 28., 29., 26., 30., 27., 32., 31.]])


canvas[:, indices[threadidx]]  <-- feature[:, threadidx]
'''


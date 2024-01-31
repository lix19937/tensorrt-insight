#
# lix  
#

import torch
# from mmcv.ops.point_sample import bilinear_grid_sample
from utils import bilinear_grid_sample

import numpy as np
from loguru import logger

def savetensor_byrow(x, file_name, fmt = "%.6f", delimiter=" "):
  shape = x.shape
  leng = len(shape)
  if leng == 1:
    x = x.reshape(1, -1)
    shape = x.shape
    leng = len(shape)
  
  flg = '-'
  b = [str(i) for i in shape] 
  shape_flg = '.'+flg.join(b)

  if leng <= 0:
    return
  if leng == 2:
    np.savetxt(file_name + shape_flg, x, fmt=fmt, delimiter=delimiter)   
  if leng > 2:
    cs = 1
    for i in range(leng - 2):
      cs = cs*shape[i]

    new_shape = (cs, shape[-2], shape[-1])
    rx = x.reshape(new_shape)
    with open(file_name + shape_flg, 'w') as f:
      for i in range(new_shape[0]):
        np.savetxt(f, rx[i], fmt=fmt, delimiter=delimiter)

## all input are fp32
def feature_sampling_onnx(mlvl_feats, reference_points, pc_range, img_shape, lidar2img):
    lidar2img = lidar2img.type_as(mlvl_feats[0])
    
    reference_points_3d = reference_points.clone()

    reference_points[..., 0:1] = reference_points[..., 0:1]*(pc_range[3] - pc_range[0]) + pc_range[0]
    reference_points[..., 1:2] = reference_points[..., 1:2]*(pc_range[4] - pc_range[1]) + pc_range[1]
    reference_points[..., 2:3] = reference_points[..., 2:3]*(pc_range[5] - pc_range[2]) + pc_range[2]


    reference_points = torch.cat((reference_points, torch.ones_like(reference_points[..., :1])), -1)
    B, num_query = reference_points.size()[:2]
    num_cam = lidar2img.size(1)
    reference_points = reference_points.view(-1, 4).permute(1, 0)
    logger.info("reference_points:{}".format(reference_points.shape))  

    ## 
    img_shapes = torch.tensor([[img_shape[0][1]], [img_shape[0][0]], [1.], [1.]])
 
    lidar2img = lidar2img/img_shapes  ## div first
    lidar2img = lidar2img.view(-1, 4) 

    logger.info("lidar2img:{}".format(lidar2img.shape))  

    ## lidar2img / img_shapes     [NC*4, 4] * [4, L] --> [NC*4, L] --> [1, NC, 4, L] 
    reference_points_cam = torch.matmul(lidar2img, reference_points) 
    logger.info("reference_points_cam:{}".format(reference_points_cam.size())) 
    reference_points_cam = reference_points_cam.view(1, num_cam, 4, num_query).permute(0, 1, 3, 2)  # --> [1, NC, L, 4]
    savetensor_byrow(reference_points_cam, 'reference_points_cam_s2_v2.data', fmt = "%.6f")
    
    ####### step2 end #################
    mask = reference_points_cam[..., 2:3] > 1e-3  
    
    reference_points_cam = torch.clamp(
                          torch.where(mask, 
                                  reference_points_cam[..., 0:2]/reference_points_cam[..., 2:3], 
                                  mask.new_tensor(torch.ones_like(reference_points_cam[..., 0:2]))*(-1.)),# -1                                  
                          min=-1., max=2.) 
    
    reference_points_cam += reference_points_cam - 1.
    # mask = (mask & (reference_points_cam[..., 0:1] > -1.0) # [1, 6, 512, 1]
    #              & (reference_points_cam[..., 0:1] < 1.0) 
    #              & (reference_points_cam[..., 1:2] > -1.0) # [1, 6, 512, 1] 
    #              & (reference_points_cam[..., 1:2] < 1.0))

    mask = (mask & (np.fabs(reference_points_cam[..., 0:1]) <1.0)            
                 & (np.fabs(reference_points_cam[..., 1:2]) <1.0))

    logger.info("mask:{}".format(mask.size()))  # [1, 6, 512, 1]   #[1, NC, L, 1]
             
    mask = mask.view(B, num_cam, 1, num_query, 1, 1).permute(0, 2, 3, 1, 4, 5) ## last step 

    reference_points_cam_lvl = reference_points_cam.view(B*num_cam, num_query, 1, 2)
    sampled_feats = []
    for _, feat in enumerate(mlvl_feats):
        B, N, C, H, W = feat.size()
        logger.info("~~~>>:{}".format(feat.size()))  # B, N, C:  1, 6, 256

        feat = feat.view(B*N, C, H, W)  
        # sampled_feat = F.grid_sample(feat, reference_points_cam_lvl)
        sampled_feat = bilinear_grid_sample_modi_v1(feat, reference_points_cam_lvl)
        logger.info("-->>:{}".format(sampled_feat.shape))  #  6, 256, 512, 1

        sampled_feat = sampled_feat.view(B, N, C, num_query, 1).permute(0, 2, 3, 1, 4)
        sampled_feats.append(sampled_feat)
    sampled_feats = torch.stack(sampled_feats, -1)
    sampled_feats = sampled_feats.view(B, C, num_query, num_cam,  1, len(mlvl_feats))
    logger.info("++>>:{}".format(sampled_feats.shape))  #  

    savetensor_byrow(sampled_feats, str("OUT_") +'sampled_feats.data', fmt = "%.6f")
    savetensor_byrow(mask.contiguous(), str("OUT_") +'mask.data', fmt = "%.6f")

    return reference_points_3d, sampled_feats, mask
	

class PYTORCH_FS(torch.nn.Module):
    def __init__(self):
        super(PYTORCH_FS, self).__init__()

    def forward(self, mlvl_feats1, mlvl_feats2, mlvl_feats3, mlvl_feats4, reference_points, pc_range, img_shape, lidar2img):
      mlvl_feats = [mlvl_feats1, mlvl_feats2, mlvl_feats3, mlvl_feats4]
      return feature_sampling_onnx_GT(mlvl_feats, reference_points, pc_range, img_shape, lidar2img)


def helper_onnx(model_file):
    import onnx, onnxsim
    shapes_onnx_filename = model_file + "_s.onnx"
    model = onnx.load(model_file)
    model_simp, check = onnxsim.simplify(model)
    assert check, "Simplified ONNX model could not be validated"
    onnx.save(model_simp, shapes_onnx_filename)


def run_onnx(mlvl_feats, reference_points, pc_range, img_shape, lidar2img, model, out_onnx):
    onnx_path = out_onnx
    with torch.no_grad():
      torch.onnx.export(model,
                        (mlvl_feats[0], mlvl_feats[1],mlvl_feats[2],mlvl_feats[3],reference_points, pc_range, img_shape, lidar2img),
                        onnx_path,
                        verbose=False,
                        opset_version=11,
                        enable_onnx_checker=True,
                        do_constant_folding=True)
    logger.info("export done")
    helper_onnx(onnx_path)
    logger.info("helper_onnx done")


def main():
    rand_seed = 123456
    np.random.seed(rand_seed)
    torch.manual_seed(rand_seed)
    torch.cuda.manual_seed_all(rand_seed)
    pc_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
    v0=torch.rand(1,6,256,36,60)
    v1=torch.rand(1,6,256,18,30)
    v2=torch.rand(1,6,256,9,15)
    v3=torch.rand(1,6,256,5,8)

    mlvl_feats=[v0,v1,v2,v3]
    num_query = 512
    num_cam = 6
    reference_points = torch.rand(1,num_query,3)

    lidar2img = torch.rand(1,num_cam,4,4)
    img_shape = torch.tensor([[288,480]]) # (1,2)  LongTensor

    logger.info("start ... {},{}".format(img_shape.shape, img_shape.type() ))
    # savetensor_byrow(reference_points, str("In_") +'reference_points.data', fmt = "%.6f")
    # savetensor_byrow(lidar2img, str("Attri_") +'lidar2img.data', fmt = "%.6f")
    # savetensor_byrow(reference_points, str("Attri_") +'pc_range.data', fmt = "%.6f")
    # savetensor_byrow(v0, str("In_") + str(0) + '_feat.data', fmt = "%.6f")
    # savetensor_byrow(v1, str("In_") + str(1) + '_feat.data', fmt = "%.6f")
    # savetensor_byrow(v2, str("In_") + str(2) + '_feat.data', fmt = "%.6f")
    # savetensor_byrow(v3, str("In_") + str(3) + '_feat.data', fmt = "%.6f")
    # savetensor_byrow(img_shape, str("In_")  + 'img_shape.data', fmt = "%d")

    feature_sampling_onnx(mlvl_feats, reference_points, pc_range, img_shape, lidar2img)
    logger.info("done")

if __name__ == '__main__':
  main()

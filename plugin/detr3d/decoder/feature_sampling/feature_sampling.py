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
    logger.info("reference_points:{}".format(reference_points.shape))  # [1, 512, 4]
    savetensor_byrow(reference_points, 'reference_points_s1.data', fmt = "%.6f")
    savetensor_byrow(reference_points.permute(0,2,1).contiguous(), 'reference_points_s1_tr.data', fmt = "%.6f")

  ####### step1 end #################

    B, num_query = reference_points.size()[:2]
    logger.info("B:{}, num_query:{}".format(B, num_query))  # B=1, num_query=512
  
    num_cam = lidar2img.size(1)
    logger.info("num_cam:{}, lidar2img:{}".format(num_cam, lidar2img.size()))  # num_cam:6, [1, 6, 4, 4]

    ##  [1, 512, 4] --->  [1, 1, 512, 4] ---> [1, 6, 512, 4]
    reference_points = reference_points.view(B, 1, num_query, 4).repeat(1, num_cam, 1, 1).unsqueeze(-1)
    logger.info("reference_points:{}".format(reference_points.size()))  # [1, 6, 512, 4, 1]

    ##  [1, 6, 4, 4]  ---> [1, 6, 1, 4, 4] --->  [1, 6, 512, 4, 4]   write dead code !!!
    lidar2img = lidar2img.view(B, num_cam, 1, 4, 4).repeat(1, 1, num_query, 1, 1)
    logger.info("lidar2img:{}".format(lidar2img.size()))  # [1, 6, 512, 4, 4]

    ## 
    # debug_tmp=lidar2img.new_tensor([img_shape[0][1], img_shape[0][0], 1, 1])[None, None, None, :]
    # logger.info("debug_tmp:{}".format(debug_tmp))  # [1, 1, 1, 4]   value: 480., 288.,   1.,   1.   
    ## write dead code !!!
    img_shapes = lidar2img.new_tensor([img_shape[0][1], img_shape[0][0], 1, 1])[None, None, None, :].repeat(B, num_cam, 1, 1) 
    logger.info("img_shapes:{}".format(img_shapes.size()))  # [1, 6, 1, 4]
    logger.info("img_shapes:{}".format(img_shapes))  # [1, 6, 1, 4]

    logger.info("lidar2img:{}".format(lidar2img.size()))  # [1, 6, 512, 4, 4]
   
    ## lidar2img / img_shapes
    ##  [1, 6, 512, 4, 4] *  [1, 6, 512, 4, 1] --> [1, 6, 512, 4, 1] --> [1, 6, 512, 4] / [1, 6, 1, 4]
    reference_points_cam = torch.matmul(lidar2img, reference_points).squeeze(-1) / img_shapes
    logger.info("reference_points_cam:{}".format(reference_points_cam.size()))  # [1, 6, 512, 4]
    savetensor_byrow(reference_points_cam, 'reference_points_cam_s2.data', fmt = "%.6f")
    savetensor_byrow(reference_points_cam.permute(0, 1, 3, 2).contiguous(), 'reference_points_cam_s2_permute.data', fmt = "%.6f")
    #return
  ####### step2 end #################

    ## for cpp  [NC*4, 4] * [4, L] -->  [NC*4, L] -> [1, NC, 4, L]
    ## cpp   reference_points_cam  [1, NC, 4, L]  -->  map to [:,:,2:3,:]^T
    ## torch reference_points_cam  [1, NC, L, 4]      [..., 2:3]
    # debug_tmp=reference_points_cam[..., 2:3]
    # logger.info("debug_tmp:{}".format(debug_tmp.shape))  # [1, 6, 512, 1]  
    
    mask = reference_points_cam[..., 2:3] > 1e-3  #[0:1]  [1:2]  [2:3]
    logger.info("mask:{}".format(mask.size()))  # [1, 6, 512, 1]
  
    logger.info("reference_points_cam:{}".format(reference_points_cam[..., 0:2].shape))  # [1, 6, 512, 2]
    logger.info("reference_points_cam:{}".format(reference_points_cam[..., 2:3].shape))  # [1, 6, 512, 1]
    
    reference_points_cam = torch.clamp(
                          torch.where(mask, 
                                  reference_points_cam[..., 0:2]/reference_points_cam[..., 2:3], 
                                  mask.new_tensor(torch.ones_like(reference_points_cam[..., 0:2]))*(-1.)),# -1
                          min=-1., max=2.)
    
    logger.info("reference_points_cam:{}".format(reference_points_cam.shape))  # [1, 6, 512, 2]
    
    reference_points_cam = (reference_points_cam - 0.5) * 2
    mask = (mask & (reference_points_cam[..., 0:1] > -1.0) # [1, 6, 512, 1]
                 & (reference_points_cam[..., 0:1] < 1.0) 
                 & (reference_points_cam[..., 1:2] > -1.0) # [1, 6, 512, 1] 
                 & (reference_points_cam[..., 1:2] < 1.0))
    logger.info("mask:{}".format(mask.size()))  # [1, 6, 512, 1]   #[1, NC, L, 1]
             
    mask = mask.view(B, num_cam, 1, num_query, 1, 1).permute(0, 2, 3, 1, 4, 5) ## last step 
    logger.info("mask:{}".format(mask.size()))  # [1, 1, 512, 6, 1, 1]
    savetensor_byrow(mask.contiguous(), 'Out_mask.data', fmt = "%.6f")
    savetensor_byrow(reference_points_cam.contiguous(), 'reference_points_cam_after_2th_norm.data', fmt = "%.6f")
    ####### step3 end #################

    sampled_feats = []

    print('mlvl_feats sz:{}'.format(len(mlvl_feats)))
    for lvl, feat in enumerate(mlvl_feats):
        B, N, C, H, W = feat.size()
        feat = feat.view(B*N, C, H, W)
        reference_points_cam_lvl = reference_points_cam.view(B*N, num_query, 1, 2)# [6, 512, 1, 2]
        # sampled_feat = F.grid_sample(feat, reference_points_cam_lvl)
        logger.info("[IN]  feat:{},{}".format(type(feat), feat.shape)) # [6, 256, 5, 8]
        logger.info("[IN]  reference_points_cam_lvl:{},{}".format(type(reference_points_cam_lvl), reference_points_cam_lvl.shape))# [6, 512, 1, 2]
        
        # savetensor_byrow(feat, str(lvl) + '_feat.data', fmt = "%.6f")
        savetensor_byrow(reference_points_cam_lvl,  str(lvl) + '_reference_points_cam_lvl.data', fmt = "%.6f")
        sampled_feat = bilinear_grid_sample(feat, reference_points_cam_lvl)###############last 
        # savetensor_byrow(sampled_feat, str(lvl) +'_sampled_feat_out.data', fmt = "%.6f")
        logger.info("[OUT] sampled_feat:{},{}".format(type(sampled_feat), sampled_feat.shape))# [6, 256, 512, 1]
        sampled_feat = sampled_feat.view(B, N, C, num_query, 1).permute(0, 2, 3, 1, 4)#      # [1, 6, 256, 512, 1]
        logger.info("[OUT] sampled_feat:{},{}".format(type(sampled_feat), sampled_feat.shape))# [1, 256, 512, 6, 1]
        sampled_feats.append(sampled_feat)
        
    logger.info('========{}, {}'.format(len(sampled_feats),sampled_feats[0].shape))# [1, 256, 512, 6, 1]
    sampled_feats = torch.stack(sampled_feats, -1)
    logger.info('========{}'.format(sampled_feats.shape))# [1, 256, 512, 6, 1, 4]

    sampled_feats = sampled_feats.view(B, C, num_query, num_cam,  1, len(mlvl_feats))
    logger.info('========{},{},{}'.format(reference_points_3d.shape, sampled_feats.shape, mask.shape))
    # torch.Size([1, 512, 3]),torch.Size([1, 256, 512, 6, 1, 4]),torch.Size([1, 1, 512, 6, 1, 1])

    savetensor_byrow(sampled_feats, 'Out_sampled_feat.data', fmt = "%.6f")

    return reference_points_3d, sampled_feats, mask

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
    savetensor_byrow(reference_points, str("In_") +'reference_points.data', fmt = "%.6f")
    savetensor_byrow(lidar2img, str("Attri_") +'lidar2img.data', fmt = "%.6f")

    savetensor_byrow(v0, str("In_") + str(0) + '_feat.data', fmt = "%.6f")
    savetensor_byrow(v1, str("In_") + str(1) + '_feat.data', fmt = "%.6f")
    savetensor_byrow(v2, str("In_") + str(2) + '_feat.data', fmt = "%.6f")
    savetensor_byrow(v3, str("In_") + str(3) + '_feat.data', fmt = "%.6f")

    savetensor_byrow(img_shape, str("In_")  + 'img_shape.data', fmt = "%d")

    feature_sampling_onnx(mlvl_feats, reference_points, pc_range, img_shape, lidar2img)
    logger.info("done")

if __name__ == '__main__':
  main()

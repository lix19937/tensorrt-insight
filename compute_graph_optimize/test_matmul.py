
import torch 
from loguru import logger
import random
import numpy as np

def seed_everything(seed=11):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(1)
bs =1
num_cams = 3
num_query = 2
D=2

def equal_op(lidar2img, reference_points):
    lidar2img = lidar2img.view(-1, 4)                             # [NC*4, 4]
    reference_points = reference_points.view(-1, 4).permute(1, 0) # [4,    D*L]

    reference_points_cam = torch.matmul(lidar2img, reference_points) # --> [NC*4, D*L]
    # reference_points_cam = reference_points_cam.view(num_cams, D, 4, num_query).permute(0, 1, 3, 2)   
    # return reference_points_cam.reshape(D, 1, num_cams, num_query, 4)
    # logger.info(reference_points_cam) # [D, 1, NC, L, 4]

    # [NC*4, D*L] -> [D*L, NC*4]
    # (NC, 4, D, L) by (2,3,0,1)  
    
    # t = reference_points_cam.permute(1, 0).reshape(D, -1, num_cams, 4).permute(0, 2, 1, 3)   # ok 
    # t = reference_points_cam.reshape(num_cams, 4, D, -1).permute(2, 3, 0, 1).permute(0, 2, 1, 3) # ok 

    t = reference_points_cam.reshape(num_cams, 4, D, -1).permute(2, 0, 3, 1).reshape(D, 1, num_cams, -1, 4) 
    logger.info(t) 
    return t


def bevformer():
    lidar2img       =torch.randn(num_cams, 4, 4).float()      # [NC, 4, 4]
    reference_points=torch.randn(bs, D, num_query, 4).float() # [1, D, L, 4]

    ref = equal_op(lidar2img, reference_points)

    lidar2img_r        = lidar2img.view(1, bs, num_cams, 1, 4, 4).repeat(D, 1, 1, num_query, 1, 1)                # [D, 1, NC, L, 4, 4]
    reference_points_r = reference_points.view(D, bs, 1, num_query, 4).repeat(1, 1, num_cams, 1, 1).unsqueeze(-1) # [D, 1, NC, L, 4, 1]
    reference_points_cam_r = torch.matmul(lidar2img_r, reference_points_r).squeeze(-1)

    logger.info(reference_points_cam_r.shape) # [D, 1, NC, L, 4]
    logger.info(reference_points_cam_r)        

    diff = reference_points_cam_r - ref 
    logger.info(diff < 1.e-6)  
                                            

def detr3d():
    lidar2img       =torch.randn(bs, num_cams, 4, 4).float() # [1, NC, 4, 4]
    reference_points=torch.randn(bs, num_query, 4).float()   # [1, L, 4]
    
    lidar2img_r = lidar2img.view(bs, num_cams, 1, 4, 4).repeat(1, 1, num_query, 1, 1)
    reference_points_r = reference_points.view(bs, 1, num_query, 4).repeat(1, num_cams, 1, 1).unsqueeze(-1)
    reference_points_cam_r = torch.matmul(lidar2img_r, reference_points_r).squeeze(-1)
    logger.info("reference_points_r:{}".format(reference_points_r.size()))  # [1, NC, L, 4, 1]
    logger.info("lidar2img_r:{}".format(lidar2img_r.size()))                # [1, NC, L, 4, 4]
    logger.info("reference_points_cam_r:{}".format(reference_points_cam_r.size())) 

    ####### math equal solution #######
    lidar2img = lidar2img.view(-1, 4)                             # [NC*4, 4]
    reference_points = reference_points.view(-1, 4).permute(1, 0) # [4,    L]
    reference_points_cam = torch.matmul(lidar2img, reference_points) 
    reference_points_cam = reference_points_cam.view(1, num_cams, 4, num_query).permute(0, 1, 3, 2)
    logger.info("reference_points_cam:{}".format(reference_points_cam.size())) 
    diff = reference_points_cam - reference_points_cam_r 
    logger.info(diff < 1.e-6)  


bevformer()

class MSDA(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, query):         
        return query.reshape(num_cams, 4, D, -1).permute(2, 3, 0, 1).permute(0, 2, 1, 3)

# [NC*4, D*L]
NC = 6; L = 40000; D=4

query = torch.randn(NC*4, D*L)  
model = MSDA().eval()

torch.onnx.export(model, (query), 
                          'permute.onnx',
                          export_params=True,
                          keep_initializers_as_inputs=True,
                          do_constant_folding=True,
                          verbose=True,
                          opset_version=13) 

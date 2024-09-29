
import torch 
from loguru import logger

def t1(reference_points, lidar2img):
  reference_points = reference_points.permute(1, 0, 2, 3)
  D, B, num_query = reference_points.size()[:3]
  num_cam = lidar2img.size(1)

  reference_points = reference_points.view(D, B, 1, num_query, 4).repeat(1, 1, num_cam, 1, 1).unsqueeze(-1)

  lidar2img = lidar2img.view(1, B, num_cam, 1, 4, 4).repeat(D, 1, 1, num_query, 1, 1)

  reference_points_cam = torch.matmul(lidar2img.to(torch.float32),
                                      reference_points.to(torch.float32)).squeeze(-1)
  logger.info(f"lidar2img , {lidar2img.shape}") # [4, 1, 6, 40000, 4, 4]
  logger.info(f"reference_points , {reference_points.shape}") # [4, 1, 6, 40000, 4, 1]
  return reference_points_cam


def t2(reference_points, lidar2img):
  num_cams = 6; D = 4
  lidar2img = lidar2img.view(-1, 4)                             # [NC*4, 4]
  reference_points = reference_points.view(-1, 4).permute(1, 0) # [4,    D*L]
  reference_points_cam = torch.matmul(lidar2img, reference_points) # --> [NC*4, D*L]
  logger.info(f"lidar2img , {lidar2img.shape}")               # [24, 4]
  logger.info(f"reference_points , {reference_points.shape}") # [4, 160000]

  reference_points_cam = reference_points_cam.reshape(num_cams, 4, D, -1).permute(2, 0, 3, 1).reshape(D, 1, num_cams, -1, 4) 
  return reference_points_cam  


##################
reference_points = torch.randn(1, 4, 40000, 4, dtype=torch.float32)
lidar2img =  torch.randn(1, 6, 4, 4, dtype=torch.float32)

a = t1(reference_points, lidar2img)
b = t2(reference_points, lidar2img)

r = torch.allclose(a, b, rtol=1.e-6, atol=1.e-6)
logger.info(r)

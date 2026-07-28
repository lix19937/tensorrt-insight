# /**************************************************************
#  * @Copyright: 2021-2022 Copyright SAIC
#  * @Author: lijinwen
#  * @Date: 2022-03-20 11:09:48
#  * @Last Modified by: lijinwen
#  * @Last Modified time: 2022-03-20 11:09:48
#  **************************************************************/

from torch import nn
import torch
import onnx
from loguru import logger
import numpy as np

print('onnx version:', onnx.__version__)

OUT_C=64; IN_C=5; P=40000; C=20
OUT_C=4; IN_C=2; P=3; C=4

def to_numpy(tensor):
  return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

def original(x, wei, bias):
  conv1 = nn.Conv2d(OUT_C, IN_C, kernel_size=1, stride=1, padding=0, bias=True)
  conv1.weight=torch.nn.Parameter(wei)  
  conv1.bias=torch.nn.Parameter(bias)                   
  logger.info("{} {}".format(type(conv1), conv1.weight.shape))
  
  out = conv1(x)
  out = torch.nn.functional.relu(out)

  logger.debug("{} {}".format(type(out), out.shape))
  return out

def optimize(x, wei, bias):
  conv1 = nn.Conv2d(OUT_C, IN_C, kernel_size=1, stride=1, padding=0, bias=True)
  conv1.weight=torch.nn.Parameter(wei)       
  conv1.bias=torch.nn.Parameter(bias) 
  logger.info("{} {}".format(type(conv1), conv1.weight.shape))

  # begin to different 
  X = x.contiguous().permute(0, 1, 3, 2).contiguous()
  logger.info("{} {}".format(type(x), x.shape))
              
  out = conv1(X)
  out = torch.nn.functional.relu(out)

  logger.warning("{} {}".format(type(out), out.shape))
  return out

def main():
  shape = (1, IN_C, C, P)
  x = torch.randn(shape).float()
  
  # NOTE: kernel size is 1*1 !!!
  shape = (OUT_C, IN_C, 1, 1)
  wei = torch.randn(shape).float() 
  
  shape = (OUT_C, )
  bias = torch.randn(shape).float() 

  #
  res_original = original(x, wei, bias)
  res_optimi = optimize(x, wei, bias).permute(0, 1, 3, 2)

  logger.info("{}".format(res_original))
  logger.info("{}".format(res_optimi))

  #=========================================================
  ret = torch.equal(res_original, res_optimi)
  logger.info("equal {}".format(ret))

  logger.info("assert_almost_equal ...")
  np.testing.assert_almost_equal(to_numpy(res_original), to_numpy(res_optimi), decimal=5) #  
  logger.info("assert_almost_equal done ")


if __name__ == "__main__":
    main()



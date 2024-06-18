

import torch
from loguru import logger 
from tensor_warper import pt
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


class SpatialCrossAttention():
    def __init__(self,
                 embed_dims=256,
                 num_cams=6):
        super(SpatialCrossAttention, self).__init__()

        self.embed_dims = embed_dims
        self.num_cams = num_cams
        self.max_len = 10000 

    def forward(self, 
                query,
                key, # not used 
                value,
                residual=None, # none 
                query_pos=None,# none 
                key_padding_mask=None,
                reference_points=None,
                spatial_shapes=None,
                reference_points_cam=None,
                bev_mask=None,
                level_start_index=None,
                flag='encoder',
                **kwargs):
        
        logger.info(pt("key", key)) # [6, 1336, 1, 256]
        logger.info(pt("value", value)) # [6, 1336, 1, 256]
        logger.info(pt("residual", residual)) #  None
        logger.info(pt("query_pos", query_pos)) #  None

        inp_residual = query
        #
        
        value = value.view(self.num_cams, -1, self.embed_dims)
           
        # plugin-A  
        #  --IN--- 
        # query                   [bs, num_query, embed_dims]        float32
        # reference_points_cam    [num_cams, bs, num_query, D, 2]    float32
        # bev_mask                [num_cams, bs, num_query, 4]       int32/int8
        #
        #  --OUT--- 
        # queries_rebatch           [bs*num_cams, max_len, embed_dims] float32
        # reference_points_rebatch  [bs*num_cams, max_len, D, 2]       float32
        # count_norm                [bs, num_query, 1]                 float32
        # indexes                   [num_cams, max_len]                int32
        
        queries_rebatch, reference_points_rebatch, count_norm, indexes = rebatch(query, reference_points_cam, bev_mask, self.max_len)

        queries = self.deformable_attention(query=queries_rebatch, 
                                            key=key, 
                                            value=value,
                                            reference_points=reference_points_rebatch, 
                                            spatial_shapes=spatial_shapes,
                                            level_start_index=level_start_index).view(1, self.num_cams, self.max_len, self.embed_dims)

        # plugin-B  
        #  --IN--- 
        # queries    [bs, num_cams, max_len, embed_dims] float32
        # count_norm [bs, num_query, 1]                  float32
        # indexes    [num_cams, max_len]                 int32
        #
        #  --OUT--
        # slots      [bs, num_query, embed_dims]         float32
        #
        slots = slots_update(queries, count_norm, indexes)

        return slots  

seed_everything(seed=11)
bs = 1; num_cams = 6; max_len = 8; embed_dims = 6; D = 4; num_query = 12  

sca = SpatialCrossAttention(embed_dims=embed_dims)

residual=None
query_pos=None
key                 =torch.randn(num_cams, 16, bs, embed_dims)
value               =torch.randn(num_cams, 16, bs, embed_dims)
query               =torch.randn(bs, num_query, embed_dims)
reference_points_cam=torch.randn(num_cams, bs, num_query, D, 2)
# use fixed data  
query               =torch.arange(0, bs*num_query*embed_dims).view(bs, num_query, embed_dims).float()
reference_points_cam=torch.arange(0, num_cams*bs*num_query*D*2).view(num_cams, bs, num_query, D, 2).float()

bev_mask            =torch.zeros(num_cams, bs, num_query, 4, dtype=torch.int)

bev_mask[0, 0, 1, :] = torch.tensor([1, 0, 0, 0], dtype=torch.int)
bev_mask[0, 0, 2, :] = torch.tensor([0, 0, 1, 0], dtype=torch.int)
bev_mask[0, 0, 8, :] = torch.tensor([1, 1, 0, 0], dtype=torch.int)
bev_mask[0, 0, 9, :] = torch.tensor([1, 1, 0, 0], dtype=torch.int)

bev_mask[1, 0, 0, :] = torch.tensor([0, 0, 0, 1], dtype=torch.int)
bev_mask[1, 0, 2, :] = torch.tensor([1, 0, 0, 0], dtype=torch.int)
bev_mask[1, 0, 4, :] = torch.tensor([1, 0, 0, 0], dtype=torch.int)
bev_mask[1, 0, 6, :] = torch.tensor([1, 0, 0, 0], dtype=torch.int)

bev_mask[2, 0, 7, :] = torch.tensor([0, 0, 0, 1], dtype=torch.int)
bev_mask[3, 0, 8, :] = torch.tensor([0, 0, 0, 1], dtype=torch.int)
bev_mask[4, 0, 9, :] = torch.tensor([0, 0, 0, 1], dtype=torch.int)
bev_mask[5, 0, 10, :] = torch.tensor([0, 0, 0, 1], dtype=torch.int)

logger.info(pt("key", key)) # [6, 1336, 1, 256]
logger.info(pt("value", value)) # [6, 1336, 1, 256]
logger.info(pt("residual", residual)) #  None
logger.info(pt("query_pos", query_pos)) #  None
logger.info(pt("query", query)) # [1, 40000, 256]
logger.info(pt("reference_points_cam", reference_points_cam)) # [6, 1, 40000, 4, 2] 
logger.info(pt("bev_mask", bev_mask)) # [6, 1, 40000, 4]  

sca.forward(query,
            key, # not used 
            value,
            reference_points_cam=reference_points_cam,
            bev_mask=bev_mask)



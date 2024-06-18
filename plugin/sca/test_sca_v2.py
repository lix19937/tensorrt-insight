

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

        if key is None:
            key = query
        if value is None:
            value = key

        if residual is None:
            inp_residual = query
            slots = torch.zeros_like(query)
        if query_pos is not None:
            query = query + query_pos

        logger.info(pt("query", query)) # [1, 40000, 256]
        logger.info(pt("reference_points_cam", reference_points_cam)) # [6, 1, 40000, 4, 2] 
        logger.info(pt("bev_mask", bev_mask)) # [6, 1, 40000, 4]  

        bs, num_query, _ = query.size()

        D = reference_points_cam.size(3)
        indexes = []
        for i, mask_per_img in enumerate(bev_mask):
            index_query_per_img = mask_per_img[0].sum(-1).nonzero().squeeze(-1)
            indexes.append(index_query_per_img)
        max_len = max([len(each) for each in indexes])
        print(" >>indexes>>", indexes) #;exit()

        print(" >>max_len>>", max_len) #;exit()
        max_len = 8 #

        ## -----------------------------------------------------------
        count = bev_mask.sum(-1) > 0
        logger.info(pt("count", count)) # [6, 1, 40000]  torch.bool 
        count = count.permute(1, 2, 0).sum(-1) # [6, 1, 40000] -> [1, 40000, 6]
        logger.info(pt("count", count)) # [1, 40000]  torch.int64

        count = torch.clamp(count, min=1.0)
        logger.info(pt("slots", slots)) # [1, 40000, 256])  torch.float32
        logger.info(pt("count[..., None]", count[..., None])) # [1, 40000, 1]
        count = 1/count[..., None]
        logger.info(pt("count", count)) # [1, 40000, 1]
        ## -----------------------------------------------------------
        
        # each camera only interacts with its corresponding BEV queries. This step can  greatly save GPU memory.
        queries_rebatch = query.new_zeros(
            [bs, self.num_cams, max_len, self.embed_dims])
        reference_points_rebatch = reference_points_cam.new_zeros(
            [bs, self.num_cams, max_len, D, 2])
        
        for j in range(bs):
            for i, reference_points_per_img in enumerate(reference_points_cam):   
                index_query_per_img = indexes[i]
                queries_rebatch[j, i, :len(index_query_per_img)] = query[j, index_query_per_img]
                reference_points_rebatch[j, i, :len(index_query_per_img)] = reference_points_per_img[j, index_query_per_img]

        num_cams, l, bs, embed_dims = key.shape

        key = key.permute(2, 0, 1, 3).reshape(
            bs * self.num_cams, l, self.embed_dims)
        
        logger.info(pt("value", value)) # [6, 1336, 1, 256]  --permute--> [1, 6, 1336, 256]  --reshape--> [6, 1336, 256], can del permute 
        value = value.view(bs * self.num_cams, l, self.embed_dims)
           
        logger.info(pt("queries_rebatch", queries_rebatch)) #  [1, num_cams, max_len, embed_dims]
        logger.info(pt("reference_points_rebatch", reference_points_rebatch)) # [1, num_cams, max_len, D, 2]
        logger.info(queries_rebatch.view(num_cams, max_len, embed_dims)) #  
        logger.info(reference_points_rebatch.view( num_cams, max_len, D*2)) #  

        # queries = self.deformable_attention(query=queries_rebatch.view(bs*self.num_cams, max_len, self.embed_dims), 
        #                                     key=key, 
        #                                     value=value,
        #                                     reference_points=reference_points_rebatch.view(bs*self.num_cams, max_len, D, 2), 
        #                                     spatial_shapes=spatial_shapes,
        #                                     level_start_index=level_start_index).view(bs, self.num_cams, max_len, self.embed_dims)
        # queries = torch.randn(bs, self.num_cams, max_len, self.embed_dims )
        queries=0.1*torch.arange(0, bs*self.num_cams*max_len*self.embed_dims).view(bs, self.num_cams, max_len, self.embed_dims).float()

        logger.info(pt("queries", queries)) # [1, 6, max_len, 256] 
        logger.info(pt("slots", slots)) # [1, 40000, 256])  torch.float32

        for j in range(bs):
            for i, index_query_per_img in enumerate(indexes):
                slots[j, index_query_per_img] += queries[j, i, :len(index_query_per_img)]

        slots = slots * count 
        logger.info(pt("slots", slots)) # [1, 40000, 256]
        logger.info(slots) # [1, 40000, 256]

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



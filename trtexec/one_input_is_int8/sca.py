
# export an onnx with one input is int8, others are fp32   


import torch 

# --IN---
# query                     [bs, num_query, embed_dims]        float32
# reference_points_cam      [num_cams, bs, num_query, D, 2]    float32
# bev_mask                  [num_cams, bs, num_query, 4]       int8-p4
#
#  --OUT---
# indexes                   [num_cams, max_len]                int32
# indexes_len               [num_cams]                         int32
# queries_rebatch           [bs*num_cams, max_len, embed_dims] float32
# reference_points_rebatch  [bs*num_cams, max_len, D, 2]       float32
# count_norm                [bs, num_query, 1]                 float32

class SCAFunction1(torch.autograd.Function):
    @staticmethod
    def forward(ctx,  
                query, 
                reference_points_cam, 
                bev_mask, 
                num_query, 
                max_len, 
                num_cams, 
                embed_dims):

        indexes     = torch.randint(0, 10000, (num_cams, max_len), dtype=torch.int32)
        indexes_len = torch.randint(1, 10000, (num_cams,), dtype=torch.int32)
        queries_rebatch          = torch.randn(bs*num_cams, max_len, embed_dims)
        reference_points_rebatch = torch.randn(bs*num_cams, max_len, 4, 2)
        count_norm               = torch.randn(bs, num_query, 1)
        
        return indexes, indexes_len, queries_rebatch, reference_points_rebatch, count_norm

    @staticmethod
    def symbolic(
        g,
        query, 
        reference_points_cam, 
        bev_mask, 
        num_query, 
        max_len, 
        num_cams, 
        embed_dims
        ):
        return g.op(
            "TRT::SCA_IndexRebatch_TRT",
            query, 
            reference_points_cam, 
            bev_mask, 
            num_query_i=num_query,
            max_len_i=max_len,
            num_cams_i=num_cams,
            embed_dims_i=embed_dims,
            outputs=5
            )


    
sca_fun_1 = SCAFunction1.apply

class MSDA(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, 
                query, 
                reference_points_cam,
                bev_mask):
        
        num_query = 40000 
        max_len = 10000 
        num_cams = 6
        embed_dims = 256
    
        indexes, indexes_len, queries_rebatch, reference_points_rebatch, count_norm = sca_fun_1(
            query, reference_points_cam, bev_mask, 
            num_query, 
            max_len, 
            num_cams, 
            embed_dims)
        return indexes, indexes_len, queries_rebatch, reference_points_rebatch, count_norm
          

# 1336 from fpn (related to img shape )
# 2304 / 9216 from bev shape 
num_cams= 6; bs =1; num_query = 40000; embed_dims=256

query =torch.randn(bs, num_query, embed_dims)  
reference_points_cam =torch.randn(num_cams, bs, num_query, 4, 2)  
bev_mask =(torch.randn(num_cams, bs, num_query, 4) > 0.5).to(dtype=torch.int8) # int8

# query                   [bs, num_query, embed_dims]        float32
# reference_points_cam    [num_cams, bs, num_query, D, 2]    float32
# bev_mask                [num_cams, bs, num_query, 4]       int32/int8

model = MSDA().eval()

torch.onnx.export(model, (
                        query, 
                        reference_points_cam,
                        bev_mask), 
                          'sca.onnx',
                          input_names=("query", "reference_points_cam", "bev_mask"),
                          export_params=True,
                          keep_initializers_as_inputs=True,
                          do_constant_folding=True,
                          opset_version=13) 


##  trtexec --onnx=./sca.onnx --plugins=./libplugin_custom.so --verbose --inputIOFormats=fp32:chw,fp32:chw,int8:chw   --outputIOFormats=int32:chw,int32:chw,fp32:chw,fp32:chw,fp32:chw

##  trtexec --onnx=./sca.onnx --plugins=./libplugin_custom.so --verbose --fp16 --inputIOFormats=fp16:chw,fp16:chw,int8:chw   --outputIOFormats=int32:chw,int32:chw,fp16:chw,fp16:chw,fp16:chw

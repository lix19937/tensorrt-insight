
import torch 

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

class SCAFunction1(torch.autograd.Function):
    @staticmethod
    def forward(ctx,  
               query, 
               reference_points_cam, 
               bev_mask, 
               max_len):
        queries_rebatch = torch.randn(reference_points_cam.shape[0], max_len, query.shape[2])
        reference_points_rebatch = torch.randn(reference_points_cam.shape[0], max_len, reference_points_cam.shape[-2], reference_points_cam.shape[-1])
        count_norm = torch.randn(query.shape[0], query.shape[1], 1)
        indexes = torch.randn(reference_points_cam.shape[0], max_len)
        return queries_rebatch, reference_points_rebatch, count_norm, indexes

    @staticmethod
    def symbolic(
        g,
        query, 
        reference_points_cam, 
        bev_mask, 
        max_len
        ):
        return g.op(
            "TRT::RebatchPlugin_TRT",
               query, 
               reference_points_cam, 
               bev_mask, 
               max_len_i=max_len,
               outputs=4
            )

# plugin-B  
#  --IN--- 
# queries    [bs, num_cams, max_len, embed_dims] float32
# count_norm [bs, num_query, 1]                  float32
# indexes    [num_cams, max_len]                 int32
#
#  --OUT--
# slots      [bs, num_query, embed_dims]         float32
#

class SCAFunction2(torch.autograd.Function):
    @staticmethod
    def forward(ctx, queries, count_norm, indexes):
        out = torch.randn(queries.shape[0], count_norm.shape[1], queries.shape[3])
        return out

    @staticmethod
    def symbolic(g, queries, count_norm, indexes):
        return g.op(
            "TRT::SlotsUpdatePlugin_TRT",
            queries, count_norm, indexes)

# queries_rebatch           [bs*num_cams, max_len, embed_dims]  float32
# reference_points_rebatch  [bs*num_cams, max_len, D, 2]        float32
#
# queries                   [bs, num_cams, max_len, embed_dims] float32
 
class SCAFunction3(torch.autograd.Function):
    @staticmethod
    def forward(ctx, queries_rebatch, reference_points_rebatch):
        out = torch.randn(1, queries_rebatch.shape[0], queries_rebatch.shape[1], queries_rebatch.shape[2])
        return out

    @staticmethod
    def symbolic(g, queries_rebatch, reference_points_rebatch):
        return g.op(
            "TRT::MSDAPlugin_TRT",
            queries_rebatch, reference_points_rebatch)
    
sca_fun_1 = SCAFunction1.apply
sca_fun_2 = SCAFunction2.apply
sca_fun_m = SCAFunction3.apply

class MSDA(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, 
                query, 
                reference_points_cam,
                bev_mask):
        
        queries_rebatch, reference_points_rebatch, count_norm, indexes = sca_fun_1(
               query, 
               reference_points_cam, 
               bev_mask, 
               8)
        
        queries = sca_fun_m(queries_rebatch, reference_points_rebatch)

        return sca_fun_2(queries, count_norm, indexes)

# 1336 from fpn (related to img shape )
# 2304 / 9216 from bev shape 
num_cams= 6; bs =1; num_query = 12; embed_dims=256

query =torch.randn(bs, num_query, embed_dims)  
reference_points_cam =torch.randn(num_cams, bs, num_query, 4, 2)  
bev_mask =(torch.randn(num_cams, bs, num_query, 4) > 0.5  ).int()

# query                   [bs, num_query, embed_dims]        float32
# reference_points_cam    [num_cams, bs, num_query, D, 2]    float32
# bev_mask                [num_cams, bs, num_query, 4]       int32/int8


model = MSDA().eval()

torch.onnx.export(model, (
                        query, 
                        reference_points_cam,
                        bev_mask), 
                          'sca.onnx',
                          export_params=True,
                          keep_initializers_as_inputs=True,
                          do_constant_folding=True,
                          verbose=True,
                          opset_version=13) 

##  polygraphy run  --load-inputs   \ --plugins

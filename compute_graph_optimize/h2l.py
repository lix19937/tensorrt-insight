
import torch
import torch.nn as nn
from loguru import logger 

class MSDeformableAttention3D(nn.Module):
    def __init__(self,
                 embed_dims=256,
                 num_heads=8,
                 num_levels=4,
                 num_points=8,
                 ):
        super().__init__()
        if embed_dims % num_heads != 0:
            raise ValueError(f'embed_dims must be divisible by num_heads, '
                             f'but got {embed_dims} and {num_heads}')
        self.output_proj = None

        self.embed_dims = embed_dims
        self.num_levels = num_levels
        self.num_heads = num_heads
        self.num_points = num_points
        self.sampling_offsets  = nn.Linear(embed_dims, num_heads * num_levels * num_points * 2)
        self.attention_weights = nn.Linear(embed_dims, num_heads * num_levels * num_points)
        self.value_proj = nn.Linear(embed_dims, embed_dims)
    
    def get_nnl_paras(self):
        return [self.sampling_offsets, self.attention_weights, self.value_proj]

    def forward(self, query, value):  
        bs, num_query, _ = query.shape
        bs, num_value, _ = value.shape

        value = self.value_proj(value)
        value = value.view(bs, num_value, self.num_heads, -1)

        sampling_offsets = self.sampling_offsets(query).view(
            bs, num_query, self.num_heads, self.num_levels, self.num_points, 2)
        
        attention_weights = self.attention_weights(query).view(
            bs, num_query, self.num_heads, self.num_levels * self.num_points)

        attention_weights = attention_weights.softmax(-1)
        attention_weights = attention_weights.view(bs, num_query,
                                                   self.num_heads,
                                                   self.num_levels,
                                                   self.num_points)
        return value, sampling_offsets, attention_weights


class MSDeformableAttention3D_OPT(nn.Module):
    def __init__(self,
                 embed_dims=256,
                 num_heads=8,
                 num_levels=4,
                 num_points=8,
                 nnl_paras=None
                 ):
        super().__init__()
        if embed_dims % num_heads != 0:
            raise ValueError(f'embed_dims must be divisible by num_heads, '
                             f'but got {embed_dims} and {num_heads}')
        self.output_proj = None

        self.embed_dims = embed_dims
        self.num_levels = num_levels
        self.num_heads = num_heads
        self.num_points = num_points
        if nnl_paras is None:
            self.sampling_offsets  = nn.Linear(embed_dims, num_heads * num_levels * num_points * 2)
            self.attention_weights = nn.Linear(embed_dims, num_heads * num_levels * num_points)
            self.value_proj = nn.Linear(embed_dims, embed_dims)
        else:
            self.sampling_offsets = nnl_paras[0]
            self.attention_weights = nnl_paras[1]
            self.value_proj = nnl_paras[2]

    def forward(self, query, value):  
        bs, num_query, _ = query.shape
        bs, num_value, _ = value.shape

        query = query.view(bs*num_query, -1)
        value = value.view(bs*num_value, -1)

        value = self.value_proj(value)
        value = value.view(bs, num_value, self.num_heads, -1)

        sampling_offsets = self.sampling_offsets(query).view(
            bs, num_query, self.num_heads, self.num_levels, self.num_points, 2)
        
        attention_weights = self.attention_weights(query).view(
            bs*num_query * self.num_heads, self.num_levels * self.num_points)

        attention_weights = attention_weights.softmax(-1)
        attention_weights = attention_weights.view(bs, num_query,
                                                   self.num_heads,
                                                   self.num_levels,
                                                   self.num_points)
        return value, sampling_offsets, attention_weights

bs=6
num_query=2304
num_value=1336
embed_dims=256

query = torch.randn(bs, num_query, embed_dims) 
value = torch.randn(bs, num_value, embed_dims) 

msd = MSDeformableAttention3D()
msd_opt = MSDeformableAttention3D_OPT(nnl_paras=msd.get_nnl_paras())

value_g, sampling_offsets, attention_weights = msd.forward(query, value)
value_t, sampling_offsets_t, attention_weights_t = msd_opt.forward(query, value)

print(torch.equal(value_g, value_t))
print(torch.equal(sampling_offsets, sampling_offsets_t))
print(torch.equal(attention_weights, attention_weights_t))

##############################
output_file = 'matmul.onnx' 
torch.onnx.export(
    msd,
    (query, value),
    output_file,
    keep_initializers_as_inputs=True,
    do_constant_folding=True,
    verbose=False,
    opset_version=13
)
logger.info("export done")

##############################
output_file_poly = output_file.replace(".onnx", "_poly.onnx")    

cmd = "polygraphy surgeon sanitize --fold-constants " + str(output_file)  + " -o " + output_file_poly
import subprocess
ret, val = subprocess.getstatusoutput(cmd)
logger.info(f'polygraphy returncode:{ret}\n{val}')
logger.info('polygraphy done')

##############################
##############################
output_file = 'matmul_opt.onnx' 
torch.onnx.export(
    msd,
    (query, value),
    output_file,
    keep_initializers_as_inputs=True,
    do_constant_folding=True,
    verbose=False,
    opset_version=13
)
logger.info("export done")

##############################
output_file_poly = output_file.replace(".onnx", "_poly.onnx")    

cmd = "polygraphy surgeon sanitize --fold-constants " + str(output_file)  + " -o " + output_file_poly
import subprocess
ret, val = subprocess.getstatusoutput(cmd)
logger.info(f'polygraphy returncode:{ret}\n{val}')
logger.info('polygraphy done')

#  trtexec --onnx=matmul_poly.onnx --verbose --best --separateProfileRun --useCudaGraph --dumpProfile
#  trtexec --onnx=matmul_opt_poly.onnx --verbose --best --separateProfileRun --useCudaGraph --dumpProfile 2>&1  > log

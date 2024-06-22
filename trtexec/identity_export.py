
import torch 

# --IN---
# query                     [bs, num_query, embed_dims]        float32
#
#  --OUT---
# query_out                 query.shape                        float32

class SCAFunction1(torch.autograd.Function):
    @staticmethod
    def forward(ctx, query):      
        return query

    @staticmethod
    def symbolic(
        g,
        query
        ):
        return g.op("TRT::Identity_TRT", query)

sca_fun_1 = SCAFunction1.apply

class MSDA(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, query):         
        return sca_fun_1(query)


bs = 1; num_query = 40000; embed_dims=256

query = torch.randn(bs, num_query, embed_dims)  
model = MSDA().eval()

torch.onnx.export(model, (query), 
                          'identity.onnx',
                          export_params=True,
                          keep_initializers_as_inputs=True,
                          do_constant_folding=True,
                          verbose=True,
                          opset_version=13) 


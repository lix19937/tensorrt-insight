import torch
import torch.nn as nn 

# https://github.com/NVIDIA/TensorRT/issues/2655  

# https://github.com/NVIDIA/TensorRT/tree/release/8.5/plugin/bertQKVToContextPlugin
# The yaml file says that version 3 is not supported yet.

class CustomQKVToContextPluginDynamic(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, i_mask, hidden_size, num_heads):

        return input
    @staticmethod
    def symbolic(g, input, i_mask, hidden_size, num_heads):
        return g.op("CustomQKVToContextPluginDynamic", input, i_mask, plugin_version_s='1', type_id_i=0, hidden_size_i=hidden_size, num_heads_i=num_heads, has_mask_i=True)

class MyModule(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        assert hidden_size % num_heads == 0
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.size_per_head = hidden_size // num_heads
        self.Wq = nn.Linear(self.hidden_size, self.hidden_size)
        self.Wk = nn.Linear(self.hidden_size, self.hidden_size)
        self.Wv = nn.Linear(self.hidden_size, self.hidden_size)
    def forward(self, x, i_mask):
        # shape of x (seq_len, batch_size, hidden_size)
        # shape of i_mask (batch_size)
        # output (seq_len, batch_size, hidden_size)
        Q = self.Wq(x)
        K = self.Wk(x)
        V = self.Wv(x)
        qkv = torch.cat([Q, K, V], dim=2)
        qkv = qkv.view(x.size(0), x.size(1), 3, self.num_heads, self.size_per_head)
        qkv = qkv.transpose(2, 3).contiguous().view(x.size(0), x.size(1), 3*self.hidden_size, 1, 1)
        return CustomQKVToContextPluginDynamic.apply(qkv, i_mask, self.hidden_size, self.num_heads).select(-1, 0).select(-1, 0)

model = MyModule(768, 8).cuda()#.half()
input = torch.randn(128, 2, 768).cuda()#.half()
i_mask = torch.tensor([[64], [78]], dtype=torch.int32).cuda()

from torch.onnx import OperatorExportTypes
torch.onnx.export(model, (input, i_mask), 'test.onnx', operator_export_type=OperatorExportTypes.ONNX_FALLTHROUGH, input_names=['input_0', 'input_1'], output_names=['output_0'])

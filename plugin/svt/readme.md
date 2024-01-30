
+ mha     
**注意batch在第几维**   
https://github.com/lix19937/dnn-cookbook/blob/main/ViT-pytorch/qkv2ctx_v2.py#L140     

![mha](pt_self_atten.png)      

+ ffn /mlp
```py

def gelu(x):
    cdf = 0.5 * (1.0 + torch.erf(x / torch.sqrt(2.0)))
    return x*cdf

def relu(x):
    cdf = torch.max(0, x) 
    return cdf
# 克服梯度消失的问题
# 加快训练速度

def forward(self, x):
    x = self.fc1(x)    # cutlass   gemm + bias 
    x = self.act_fn(x) # use gelu  
    x = self.fc2(x)    # cutlass   gemm + bias 
    return x
```   
![ffn](ffn-use-gelu.png)

+ ln
```py
# alpha, beta来自train 阶段得到的    
mean = torch.mean(input, dim=(2), keepdim=True)

# 使用有偏估计 ，即进行总体方差计算     
# var = sum_i(input_i - mean )**2 / n

var = torch.var(input, dim=(2), keepdim=True, unbiased=False)
 
output = alpha * (input - mean) / ((var+1e-5)**0.5) + beta
```
![ln](ln.png)

+ ca     
![ca](ca.png)


+ mha    
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
![ln](ln.png)

+ ca     
![ca](ca.png)

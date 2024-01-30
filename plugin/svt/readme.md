
+ mha    
![mha](pt_self_atten.png)     

+ ffn /mlp
```py

def gelu(x):
    cdf = 0.5 * (1.0 + tf.erf(x / tf.sqrt(2.0)))
    return x*cdf

def forward(self, x):
    x = self.fc1(x)
    x = self.act_fn(x) # use gelu  
    x = self.fc2(x)
    return x
```   
![ffn](ffn.png)

+ ln    
![ln](ln.png)

+ ca     
![ca](ca.png)

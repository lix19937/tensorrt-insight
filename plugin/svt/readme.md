
+ mha    
![mha](pt_self_atten.png)     

+ ffn /mlp
```py
def forward(self, x):
    x = self.fc1(x)
    x = self.act_fn(x)
    x = self.fc2(x)
    return x
```   
![ffn](ffn.png)

+ ln    
![ln](ln.png)

+ ca     
![ca](ca.png)

import torch   

c = a -b 
max = torch.max(c)
min = torch.min(c)

r = torch.allclose(a, b, rtol=1.e-3, atol=1.e-1)

logger.info(f"{max}, {min}")

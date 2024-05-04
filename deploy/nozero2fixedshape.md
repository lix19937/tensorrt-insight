
尝试用torch.where()以及toch.stack()代替torch.nonzero()   

```py 

# patch_q_cnt shape [b, k]
patch_nonzero_idxs = (patch_q_cnt > 0).nozero(as_tuple=False) 

# ---------------等效做法----------------  
patch_q_idxs_tmp = torch.where(patch_q_cnt > 0)
patch_nonzero_idxs = torch.stack(patch_q_idxs_tmp, dim=1)
```


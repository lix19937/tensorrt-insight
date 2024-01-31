
```
block_FIXED.query_pos  # (L, B, hidden_size or embed_dims) eg. hidden_size = 256, query_pos use in `mha, ca`
block_-1.query         # (L, B, hidden_size or embed_dims) eg. hidden_size = 256, update by loop
```

## ('MultiheadAttention', 'norm', 'SVDetCrossAtten', 'norm', 'ffn', 'norm', 'reg_update')      

## MultiheadAttention      
```
block_K.mh_attention.query.weight  # (hidden_size, hidden_size)     + query_pos  
block_K.mh_attention.query.bias    # (hidden_size, 1)
block_K.mh_attention.key.weight    # (hidden_size, hidden_size)     + query_pos  
block_K.mh_attention.key.bias      # (hidden_size, 1)
block_K.mh_attention.value.weight  # (hidden_size, hidden_size)
block_K.mh_attention.value.bias    # (hidden_size, 1)
block_K.mh_attention.out.weight    # (hidden_size, hidden_size)
block_K.mh_attention.out.bias      # (hidden_size, 1)
```

## norm     
```
block_K.mh_attention_norm.ln.weight  # (hidden_size, )
block_K.mh_attention_norm.ln.bias    # (hidden_size, )
```
## SVDetCrossAtten       
```
  + query_pos

block_K.cross_attention.attention_weights.fc.weight # (hidden_size, 24)
block_K.cross_attention.attention_weights.fc.bias   # (24, ) 

block_K.cross_attention.output_proj.fc.weight  # (hidden_size, hidden_size)
block_K.cross_attention.output_proj.fc.bias    # (hidden_size, )

block_K.cross_attention.position_encoder.fc1.weight # (3, hidden_size)
block_K.cross_attention.position_encoder.fc1.bias   # (hidden_size, )

block_K.cross_attention.position_encoder.ln1.weight  # (hidden_size, )
block_K.cross_attention.position_encoder.ln1.bias    # (hidden_size, )

block_K.cross_attention.position_encoder.fc2.weight  # (hidden_size, hidden_size)
block_K.cross_attention.position_encoder.fc2.bias    # (hidden_size, )

block_K.cross_attention.position_encoder.ln2.weight  # (hidden_size, )
block_K.cross_attention.position_encoder.ln2.bias    # (hidden_size, )
```

## norm    
```
block_K.cross_attention_norm.ln.weight  # (hidden_size, )
block_K.cross_attention_norm.ln.bias    # (hidden_size, )
```

## ffn    
```
block_K.ffn.fc1.weight  # (hidden_size, inner_num) , inner_num is L
block_K.ffn.fc1.bias    # (inner_num, )
block_K.ffn.fc2.weight  # (inner_num, hidden_size)
block_K.ffn.fc2.bias    # (hidden_size, )
```

## norm    
```
block_K.ffn_norm.ln.weight  # (hidden_size, )
block_K.ffn_norm.ln.bias    # (hidden_size, )
```

## reg_branches      
```
block_K.reg_branches.fc.weight  # (hidden_size, hidden_size)
block_K.reg_branches.fc.bias    # (hidden_size, )
block_K.reg_branches.fc.weight  # (hidden_size, hidden_size)  
block_K.reg_branches.fc.bias    # (hidden_size, )    
block_K.reg_branches.fc.weight  # (hidden_size, 8)   
block_K.reg_branches.fc.bias    # (8, )   
```

-------------------------------------------------------------

K=1,2,3,4   

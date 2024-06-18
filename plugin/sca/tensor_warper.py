import torch

def pt(msg, t):
  tstr = msg
  if t is not None:
    if torch.is_tensor(t):
      if t.numel() > 48:
          tstr += f"  {t.shape}  {t.dtype}  {t.device} {t.is_contiguous()}"  
      else:
          tstr += f"  {t.shape}  {t.dtype}  {t.device}  {t}  {t.is_contiguous()}"  
    elif isinstance(t, dict):
      tstr += f"  \n"
      for k, v in t.items():
        tstr += pt(str(k), v) + "\n"
    elif isinstance(t, list):
      tstr +=  f"  {type(t)}  {t}"          
    else:
      tstr +=  f"  {type(t)}  {t}"         
  else:  
    tstr += f"  None"       
  return tstr  
   





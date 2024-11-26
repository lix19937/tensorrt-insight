import torch

def getmm(t):
  max_value = torch.max(t)
  min_value = torch.min(t)
  return max_value.item(), min_value.item()


def pt(msg, t):
  tstr = msg
  if t is not None:
    if torch.is_tensor(t):
      if t.numel() > 48:
          max_value, min_value = getmm(t)
          tstr += f"  {t.shape}  {t.dtype}  {t.device} {t.is_contiguous()} [{min_value} {max_value}]"  
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
   

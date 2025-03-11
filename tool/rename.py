
# from char-ordered file to number-ordered file

import os
filePath = 'result_yr_best3_ego/'
dst_dir = "out/"
fs = os.listdir(filePath)
idx=0
for it in fs:
  os.rename(filePath + it,  dst_dir + str(idx) + ".jpg")
  idx +=1




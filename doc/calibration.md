## ptq   

ptq calib 期间可以进行fuse_bn，减少bn layer的标定，降低标定时间和calib 误差    

sensitivity layer profile  
*  找到所有quant layer  
*  每次仅使能一层quant layer进行指标eval，记录到dict中 {"layer_name":eval_value}       


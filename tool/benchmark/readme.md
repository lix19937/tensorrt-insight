# trt benchmark 

## Compile  

```shell  
make -j${proc}
```  

## Run  

```shell  
./test ./resnet50.engine  ./vgg.engine

```


## Note   
It has test passed on TensorRT 8.5.10, 8.6.0.1    

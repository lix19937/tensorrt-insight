```
For the generalized NC/xHWx layout format, the following observations apply:
Only the channel dimension, C, is grouped into x channels each.

When x = 1, each group has only one channel. Hence, the elements of one channel (that is, one group) are 
arranged contiguously (in the row-major order), before proceeding to the next group (that is, next 
channel). This is the same as NCHW format.   

When x = C, then NC/xHWx is identical to NHWC, that is, the entire channel depth C is considered as a 
single group. The case x = C can be thought of as vectorizing the entire C dimension as one big vector, 
laying out all the Cs, followed by the remaining dimensions, just like NHWC.

The tensor format cudnnTensorFormat_t can also be interpreted in the following way: The NCHW INT8x32 
format is really N x (C/32) x H x W x 32 (32 Cs for every W), just as the NCHW INT8x4 format is N x (C/4)
x H x W x 4 (4 Cs for every W). Hence the VECT_C name - each W is a vector (4 or 32) of Cs.

```

## kHWC8  
```
    //! Eight channel format where C is padded to a multiple of 8. This format
    //! is bound to FP16. It is only available for dimensions >= 3.
    //! For a tensor with dimensions {N, C, H, W},
    //! the memory layout is equivalent to the array with dimensions
    //! [N][H][W][(C+7)/8*8], with the tensor coordinates (n, c, h, w)
    //! mapping to array subscript [n][h][w][c].
```

## kCHW32  
``` 
    //! Thirty-two wide channel vectorized row major format. This format is
    //! only available for dimensions >= 3.
    //! For a tensor with dimensions {N, C, H, W},
    //! the memory layout is equivalent to a C array with dimensions
    //! [N][(C+31)/32][H][W][32], with the tensor coordinates (n, c, h, w)
    //! mapping to array subscript [n][c/32][h][w][c%32].
    //!
    //! For DLA usage, this format maps to the native image format for INT8,
    //! and the tensor sizes are limited to C,H,W in the range [1,8192].
```

https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#data-format-desc   
https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#reformat-free-network-tensors    
```
Note that for the vectorized formats, the channel dimension must be zero-padded to the multiple of the 
vector size. For example, if an input binding has dimensions of [16,3,224,224], kHALF data type, and kHWC8  
format, then the actual-required size of the binding buffer would be 16*8*224*224*sizeof(half) bytes, even
 though the engine->getBindingDimension() API will return tensor dimensions as [16,3,224,224]. The values 
in the padded part (that is, where C=3,4,…,7 in this example) must be filled with zeros.
```   
Refer to Data Format Descriptions for how the data are actually laid out in memory for these formats.


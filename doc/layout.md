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

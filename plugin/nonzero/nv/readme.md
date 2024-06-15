# NonZero 

Implements a plugin for the NonZero operation, customizable to output the non-zero indices in
either a row order (each set of indices in the same row) or column order format (each set of indices in the same column).

NonZero is an operation **where the non-zero indices of the input tensor is found**. 

`NonZeroPlugin` in this sample is written to handle 2-D input tensors of shape $R \times C$. Assume that the tensor contains $K$ non-zero elements and that the
non-zero indices are required in a row ordering (each set of indices in its own row). Then the output shape would be $K \times 2$.

```
outputs[0].d[1] = 2;
outputs[0].d[0] = numNonZeroSizeTensor; // K 
```

The `NonZeroPlugin` can also be configured to emit the non-zero indices in a column-order fashion through the `rowOrder` plugin attribute, by setting it to `0`.
In this case, the first output of the plugin will have shape $2 \times K$, and the output shape specification must be adjusted accordingly.


### Running inference

As sample inputs, random images from MNIST dataset are selected and scaled to between `[0,1]`. The network will output both the non-zero indices, as well as the non-zero count.

	```
	[I] Input:
	[I] 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,   0,   0,  0,    0,    0,   0
	[I] 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,   0,   0,  0,    0,    0,   0
	[I] 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,   0,   0,  0,    0,    0.8, 0
	[I] 0, 0, 0, 0, 0, 0,  0,  0,  0,  0.8, 0,   0,  0.7,  0,    0.5, 0
	[I] 0, 0, 0, 0, 0, 0,  0,  0,  0,  0.3, 0,   0,  0.4,  0,    0,   0
	[I] 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,   0,   0,  0.4,  0,    0,   0
	[I] 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,   0,   0,  0,    0,    0,   0
	[I] 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,   0,   0,  0,    0,    0,   0
	[I] 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,   0,   0,  0.8,  0,    0,   0.1
	[I] 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,   0,   0,  0.5,  0,    0,   0.9
	[I] 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,   0,   0,  0,    0,    0,   0.2
	[I] 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,   0,   0,  0,    0,    0,   0
	[I] 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,   0,   0,  0,    0,    0,   0
	[I] 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,   0,   0,  0,    0,    0,   0.8
	[I] 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,   0.8, 0,  0,    0.1,  0,   0.5
	[I] 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,   0.5, 0,  0,    0.9,  0,   0
	[I] 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,   0,   0,  0,    0.2,  0,   0
	[I]
	[I] Output:
	[I] 2  14
	[I] 3  9
	[I] 3  12
	[I] 3  14
	[I] 4  9
	[I] 4  12
	[I] 5  12
	[I] 8  12
	[I] 8  15
	[I] 9  12
	[I] 9  15
	[I] 10 15
	[I] 13 15
	[I] 14 10
	[I] 14 13
	[I] 14 15
	[I] 15 10
	[I] 15 13
	[I] 16 13
	```     

**NonZero**
- [ONNX: NonZero](https://onnx.ai/onnx/operators/onnx__NonZero.html)

**TensorRT plugins**
- [Extending TensorRT with Custom Layers](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#extending)

**Other documentation**
- [Introduction To NVIDIA’s TensorRT Samples](https://docs.nvidia.com/deeplearning/sdk/tensorrt-sample-support-guide/index.html#samples)
- [Working With TensorRT Using The C++ API](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#c_topics)
- [NVIDIA’s TensorRT Documentation Library](https://docs.nvidia.com/deeplearning/sdk/tensorrt-archived/index.html)

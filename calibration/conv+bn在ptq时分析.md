## case 1 
```py
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.bn1(x)

        x = self.mp1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.mp2(x)
        x = x.reshape(-1, 64 * 7 * 7)
        x = self.fc1(x)
        x = self.relu3(x)
        y = self.fc2(x)
        z = self.st(y)
        z = t.argmax(z, dim=1)
        return y, z
```
```
TRT-8510-EntropyCalibration2
x: 4000890a
/conv1/Conv_output_0: 3f9db09c
/relu1/Relu_output_0: 3fb6163c
/bn1/BatchNormalization_output_0: 3fb61601
/mp1/MaxPool_output_0: 3fb61601
/conv2/Conv_output_0: 3f6fc9d8
/relu2/Relu_output_0: 3f553ff1
/mp2/MaxPool_output_0: 3f553ff1
/Reshape_output_0: 3f6fc9d8
(Unnamed Layer* 8) [Constant]_output: 39137939
(Unnamed Layer* 9) [Matrix Multiply]_output: 3e74c2e0
(Unnamed Layer* 10) [Constant]_output: 39133ad3
(Unnamed Layer* 11) [Shuffle]_output: 39133ad3
/fc1/Gemm_output_0: 3e5312f6
/relu3/Relu_output_0: 3e9e7a74
(Unnamed Layer* 14) [Constant]_output: 398109de
(Unnamed Layer* 15) [Matrix Multiply]_output: 3e132edf
(Unnamed Layer* 16) [Constant]_output: 395944c4
(Unnamed Layer* 17) [Shuffle]_output: 395944c4
y: 3e1338d5
(Unnamed Layer* 25) [Softmax]_output: 3a58d0d6
(Unnamed Layer* 28) [TopK]_output_1: 3be6d87f
```

## case 2    
```py
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        # x = self.bn1(x)

        x = self.mp1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.mp2(x)
        x = x.reshape(-1, 64 * 7 * 7)
        x = self.fc1(x)
        x = self.relu3(x)
        y = self.fc2(x)
        z = self.st(y)
        z = t.argmax(z, dim=1)
        return y, z
```

```
TRT-8510-EntropyCalibration2
x: 4000890a
/conv1/Conv_output_0: 3f9db09c
/relu1/Relu_output_0: 3fb6163c
/mp1/MaxPool_output_0: 3fb6163c
/conv2/Conv_output_0: 3f6fca27
/relu2/Relu_output_0: 3f554037
/mp2/MaxPool_output_0: 3f554037
/Reshape_output_0: 3f6fca27
(Unnamed Layer* 7) [Constant]_output: 39137939
(Unnamed Layer* 8) [Matrix Multiply]_output: 3e74c32e
(Unnamed Layer* 9) [Constant]_output: 39133ad3
(Unnamed Layer* 10) [Shuffle]_output: 39133ad3
/fc1/Gemm_output_0: 3e531339
/relu3/Relu_output_0: 3e9e7aa7
(Unnamed Layer* 13) [Constant]_output: 398109de
(Unnamed Layer* 14) [Matrix Multiply]_output: 3e132f0e
(Unnamed Layer* 15) [Constant]_output: 395944c4
(Unnamed Layer* 16) [Shuffle]_output: 395944c4
y: 3e133904
(Unnamed Layer* 24) [Softmax]_output: 3a58d0e0
(Unnamed Layer* 27) [TopK]_output_1: 3be6d889
```

## case 3   
```py
    def forward(self, x):
        x = self.conv1(x)
        # x = self.relu1(x)
        x = self.bn1(x)

        x = self.mp1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.mp2(x)
        x = x.reshape(-1, 64 * 7 * 7)
        x = self.fc1(x)
        x = self.relu3(x)
        y = self.fc2(x)
        z = self.st(y)
        z = t.argmax(z, dim=1)
        return y, z
```

```
TRT-8510-EntropyCalibration2
x: 4000890a
/conv1/Conv_output_0: 3f9db069
/mp1/MaxPool_output_0: 3f9db069
/conv2/Conv_output_0: 3f3602d4
/relu2/Relu_output_0: 3f1b1f34
/mp2/MaxPool_output_0: 3f1b1f34
/Reshape_output_0: 3f33910b
(Unnamed Layer* 6) [Constant]_output: 39137939
(Unnamed Layer* 7) [Matrix Multiply]_output: 3e96eebc
(Unnamed Layer* 8) [Constant]_output: 39133ad3
(Unnamed Layer* 9) [Shuffle]_output: 39133ad3
/fc1/Gemm_output_0: 3e96f557
/relu3/Relu_output_0: 3e9a0847
(Unnamed Layer* 12) [Constant]_output: 398109de
(Unnamed Layer* 13) [Matrix Multiply]_output: 3e101b1f
(Unnamed Layer* 14) [Constant]_output: 395944c4
(Unnamed Layer* 15) [Shuffle]_output: 395944c4
y: 3e102515
(Unnamed Layer* 23) [Softmax]_output: 3a53f674
(Unnamed Layer* 26) [TopK]_output_1: 3bed87ca
```

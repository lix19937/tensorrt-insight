apex/apex/contrib/sparsity     
https://github.com/NVIDIA/apex/tree/master/apex/contrib/sparsity    commit id:b496d85fb88a801d8e680872a12822de310951fd     
### base https://github.com/NVIDIA/apex/commit/b496d85fb88a801d8e680872a12822de310951fd  版本    

-------------------------------
## 安装  
```
cd apex-master
# https://github.com/NVIDIA/apex/tree/6102d2c300fd26f4c312d5136a7b85296286c7a3    torch 2.1.0+cu121
```

在 setup.py 增加
```py
from pprint import pprint 
pprint(sys.argv)
print("done by lix19937")

if "--cpp_ext" not in sys.argv: 
    sys.argv.append('--cpp_ext')

if "--cuda_ext" not in sys.argv: 
    sys.argv.append('--cuda_ext')
```
```
pip --version
# pip 24.0   
```
if pip >= 23.1 (ref: https://pip.pypa.io/en/stable/news/#v23-1) which supports multiple `--config-settings` with the same key...     
```
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./
```
otherwise     
```
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

## 测试   
```
cd apex/contrib/sparsity  

python ./test/toy_problem.py
```

## Sparse-QAT   
可以在量化完之后 进行 稀疏，反之则不行      
 

## 稀疏原理    

NVIDIA Ampere Architecture GPUs support Structured Sparsity. To make use of this feature to achieve higher inference performance, the `convolution kernel weights` and the `fully connected weights` must meet the following requirements:     
For each output channel and for each spatial pixel in the kernel weights, every four input channels must have at least two zeros. In other words, assuming that the kernel weights have the shape [K, C, R, S] and C % 4 == 0, then the requirement is verified using the following algorithm:     

每四个输入通道必须至少有两个零。   

![kcrs](https://github.com/lix19937/tensorrt-insight/assets/38753233/ec85a73c-c704-4f30-ae78-6122a90c7991)

### weight 的布局 kcrs   
k 输出通道 ， c输入(tensor)通道， r 高度， s 宽度        
```py
hasSparseWeights = True
for k in range(0, K):
    for r in range(0, R):  # height  
        for s in range(0, S): # width  
            for c_packed in range(0, C // 4): # 通道整除4  
                // 如果非零数目大于2 了 ，那就不能稀疏  
                if numpy.count_nonzero(weights[k, c_packed*4:(c_packed+1)*4, r, s]) > 2 :
                    hasSparseWeights = False
                    
```
强制内核权重具有结构化稀疏性模式可能导致精度损失。为了通过进一步的微调来恢复丢失的精度，需要进行稀疏训练。  https://github.com/NVIDIA/apex/tree/master/apex/contrib/sparsity     


实现模型的2：4稀疏剪枝，同时还可以通过开启通道置换算法将绝对值较大的参数进行保留，以求对模型精度的影响最小化。prune_trained_model函数会计算出`稀疏mask`并将其施加在模型的权重上。

在使用ASP对一个新的（未经过稀疏的）推理模型启用结构化稀疏时需要同时调用init_model_for_pruning和compute_sparse_masks方法。  

init_model_for_pruning会为模型层添加新的mask buffer，用于保存compute_sparse_masks生成的mask，因此调用了 compute_sparse_masks后的模型的**state_dict会比之前多出一些数据**，这些数据均以_mma_mask结尾的名字进行命名。       

对于已经使用ASP enable了结构化稀疏的模型，在保存后重新加载时，需要先创建一个新的模型，并调用init_model_for_pruning方法为模型添加mask buffer后再load模型的state_dict，否则因为新模型的state_dict和之前保存的state_dict不同而报错。


该项目还可以通过开启通道置换算法，来为结构化稀疏后的模型保留最大的精度值。
通道置换算法，顾名思义，就是通过沿着权重矩阵的通道维度进行置换，并对其周围的模型层进行适当调整。




ASP(Automatic sparsity)，该模块仅仅向python模型训练文件中添加两行代码来实现模型的2：4稀疏剪枝，
同时还可以通过开启通道置换算法将绝对值较大的参数进行保留，以求对模型精度的影响最小化。


```bash  
git clone https://github.com/NVIDIA/apex.git
cd apex
git checkout 23.05
pip install -v --disable-pip-version-check --no-cache-dir \
--global-option="--cpp_ext" --global-option="--cuda_ext" \
--global-option="--permutation_search" ./
```

## 初级入门，only两步：  
```py 
# 1. 导入sparsity模块
from apex.contrib.sparsity import ASP
# 2. 使用ASP来模型和优化器进行稀疏化
ASP.prune_trained_model(model, optimizer)
```

prune_trained_model函数会计算出稀疏mask并将其施加在模型的权重上。

```py
ASP.prune_trained_model(model, optimizer)

x, y = DataLoader(args)
for epoch in range(epochs):
    y_pred = model(x)
    loss = loss_function(y_pred, y)
    loss.backward()
    optimizer.step()

torch.save(...)
```

## 非标准用法：   

ASP还可以用来为模型生成稀疏的随机化参数，从而进行更加复杂高级的实验，如果在两个step之间重新计算权重的稀疏矩阵，可以通过在训练的step之间调用ASP.recompute_sparse_masks函数来为模型重新生成稀疏mask。

### Channel Permutation
该项目还可以通过开启通道置换算法，来为结构化稀疏后的模型保留最大的精度值。

通道置换算法，顾名思义，就是通过沿着权重矩阵的通道维度进行置换，并对其周围的模型层进行适当调整。

如果开启通道置换算法，那么最终的模型精度与置换算法的质量之间存在很大关系，置换的过程可以通过Apex CUDA拓展来进行加速，否则时间会非常的久。

在 Installation 步骤中，参数--global-option="--permutation_search"即是用于安装permutation search CUDA extension 。

如果不希望开启通道置换算法，可以在ASP.init_model_for_pruning方法中将参数allow_permutation的值设置为False即可，这一点在后续的源代码分析中也会提到。

需要注意的是，当使用多个GPU时，需要为所有的GPU设置相同的随机种子，通过permutation_lib.py中的 set_identical_seed来进行设置  
```py
import torch
import numpy
import random

torch.manual_seed(identical_seed)
torch.cuda.manual_seed_all(identical_seed)
numpy.random.seed(identical_seed)
random.seed(identical_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

## Tips：
在使用ASP对一个新的（未经过稀疏的）推理模型启用结构化稀疏时需要同时调用init_model_for_pruning和compute_sparse_masks方法。   
init_model_for_pruning会为模型层添加新的mask buffer，用于保存compute_sparse_masks生成的mask，因此调用了compute_sparse_masks后的模型的state_dict会比之前多出一些数据，这些数据均以_mma_mask结尾的名字进行命名。     
对于已经使用ASP enable了结构化稀疏的模型，在保存后重新加载时，需要先创建一个新的模型，并调用init_model_for_pruning方法为模型添加mask buffer后再load模型的state_dict，否则因为新模型的state_dict和之前保存的state_dict不同而报错。 


## demo  

```py
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from apex.contrib.sparsity import ASP

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(-1, 32 * 7 * 7)
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        x = self.sig(x)
        return x

def train_loop(model, optimizer, criterion):
    num_epochs = 1
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 100 == 99:
                print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{i+1}/{len(trainloader)}], Loss: {running_loss/100:.4f}')
                running_loss = 0.0

def val(model):
    correct = 0
    total = 0

    model.eval()
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            accuracy =  correct / total * 100
            print("Test Accuracy :{}%".format(accuracy))
    return accuracy


def main():
    print('Begin to train the dense network!')
    train_loop(model, optimizer, criterion)
    print('Finish training the dense network!')
    accuracy_dense = val(model)
    print('The accuracy of the trained dense network is : {}'.format(accuracy_dense))

    torch.save(model.state_dict(), 'model_weights.pth')

    ASP.prune_trained_model(model, optimizer)
    accuracy_sparse = val(model)
    print('The accuracy of the truned  network is : {}'.format(accuracy_sparse))

    print('Begin to train the sparse network!')
    train_loop(model, optimizer, criterion)
    print('Finish training the sparse network!')
    accuracy_sparse = val(model)
    print('The accuracy of the trained sparse network is : {}'.format(accuracy_sparse))
    torch.save(model.state_dict(), 'model_weights_sparse.pth')
    print('Training finished!')


if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ConvNet().to(device)

    print('original weights has been saved!')

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    main()

# python train.py 

```

 
简单实现 https://blog.csdn.net/qq_40672115/article/details/130035270?spm=1001.2101.3001.6650.2&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7ERate-2-130035270-blog-132298127.235%5Ev43%5Epc_blog_bottom_relevance_base6&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7ERate-2-130035270-blog-132298127.235%5Ev43%5Epc_blog_bottom_relevance_base6&utm_relevant_index=3

## Ref   
apex   
https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#structured-sparsity    
https://blog.csdn.net/weixin_43669978/article/details/132298127   
https://docs.nvidia.com/deeplearning/performance/dl-performance-convolutional/index.html   

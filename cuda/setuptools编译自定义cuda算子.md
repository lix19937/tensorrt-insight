
```cc
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &correlation_forward_cuda, "Correlation forward (CUDA)");
  m.def("backward", &correlation_backward_cuda, "Correlation backward (CUDA)");
}
```

```py
import torch
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

cxx_args = ['-std=c++11']

nvcc_args = [
    '-gencode', 'arch=compute_61,code=sm_61',
    '-gencode', 'arch=compute_70,code=sm_70',
    '-gencode', 'arch=compute_70,code=compute_70'
]

setup(
    name='correlation_cuda',
    ext_modules=[
        CUDAExtension(
          name='correlation_cuda',
          sources=[
            'correlation_cuda.cc',
            'correlation_cuda_kernel.cu'
          ],
          extra_compile_args={'cxx': cxx_args, 'nvcc': nvcc_args}
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
```

python3 setup.py build

nvcc -c correlation_cuda_kernel.cu -o correlation_cuda_kernel.o

c++ -c http://correlation_cuda.cc -o correlation_cuda.o

x86_64-linux-gnu-g++ -shared correlation_cuda.o correlation_cuda_kernel.o correlation_cuda.cpython-37m-x86_64-linux-gnu.so

这里在调用.so库的时候，需要注意先import torch，再import correlation_cuda。


在量化感知训练（QAT）或量化推理中，确保 Softmax 层的浮点计算主要依赖于 PyTorch 的量化工具中对该操作的处理方式。以下是具体方法和机制，帮助您理解如何确保 Softmax 的浮点保留：

1. 保持 Softmax 的未量化状态    
在 PyTorch 中，torch.nn.Softmax 或 torch.nn.functional.softmax 默认使用浮点计算。    
在 QAT 或模型量化阶段，不会为 Softmax 自动插入假量化（Fake Quantization）模块，也不会将其替换为整数实现。

因此，Softmax 本质上不会被量化，仍然执行浮点运算。

2. 量化流程中的控制    
在量化模型时，可以通过以下方法确保 Softmax 不会被量化：    
在模型的量化准备阶段 (torch.quantization.prepare_qat 或 torch.quantization.prepare)，只对支持量化的层（如 Linear, Conv2d）插入量化模拟。   

由于 Softmax 通常位于网络的中间或输出位置，没有整数实现，其在 QAT 转换 (torch.quantization.convert) 时不会被修改。    
关键点：在量化配置过程中，Softmax 不会出现在默认的量化模块映射 (get_default_qconfig_mapping 或 get_default_qat_module_mappings) 中，因此它始终保持浮点运算。    

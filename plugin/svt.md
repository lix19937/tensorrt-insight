```
1 插件实现  
1.1 插件组成   
1.1.1 svt overview
1.1.2 decoder pipeline (mha --> mha_norm --> svca --> svca_norm --> ffn --> ffn_norm --> reg --> update | --> mha --> mha_norm ... )
1.1.3 svt dummy node in onnx (SvTransformerDecoder: 14 configuration parameter, 160 weight/bias, total 174 attributes)
1.2 kernel融合
1.2.1 纵向: 同一条数据流中操作，elementwise   
1.2.2 横向: 相互操作独立，不同数据流或同一数据流中运算数据无依赖
1.2.3 访存: 减少内存移动    
1.3 高维矩阵乘除法交换与乘法降维        
1.4 cudagraph应用  
1.5 backbone maxpool融合  
1.6 free reformat 
1.6.1 reformatting copynode的产生  
1.6.2 free reformatting的实现 
2 插件封装   
2.1 超参数据储存与加载       
2.1.1 将超参数作为插件的输入，存储到graph.initializer中，每一次迭代都作为只读参数输入给插件    
2.1.2 将超参数作为插件的属性，以const类型存储到value info中，方便大批量权重参数按统一方式设置，减少了插件的输入tensor数目，因此在svt中优先采用
2.2 运行时同时支持fp32、half和int8     
2.3 fake-int8支持    
3 插件联调
3.1 identify layer    
3.2 带插件PTQ  
4 其它
4.1 sigmoid函数加速    
4.1.1 线性逼近
4.2 backbone中slices sampling等价替换 
4.3 permute操作转换辅助函数             
4.4  拓展的torch代码
参考
```


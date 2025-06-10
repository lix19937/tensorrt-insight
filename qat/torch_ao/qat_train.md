
对于多输入  多输出    

此时 每一个输入需要 插入独立的 quant_stub， 因为每一个 输入需要进行量化       

而输出可以共用一个 dquant_stub ，因为 dquant scale来自上一个node 的输出的scale    


![img](./qat_train.png)

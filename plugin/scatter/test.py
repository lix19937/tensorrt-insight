import torch

# 1. 准备“画布” (Output)
output = torch.zeros((3, 3))

# 2. 准备“索引” (Indices)  即 坐标
indices = torch.tensor([[0, 0], [2, 1]])

# 3. 准备“更新值” (features)
features = torch.tensor([10.0, 20.0])

# 4. 执行操作 (PyTorch 中使用 index_put_)
# 需要将 indices 转置为 (row_indices, col_indices) 的形式
output.index_put_((indices[:, 0], indices[:, 1]), features)

print(output)

# tensor([[10.,  0.,  0.],
#         [ 0.,  0.,  0.],
#         [ 0., 20.,  0.]])


# 假设在一个3x3 的全零矩阵中，在坐标 (0,0) 放 10, 在坐标 (2,1) 放 20

print("-----------------------------------------")

output = torch.zeros(4, 9, dtype=torch.float32)

indices = torch.tensor([1, 0, 0, 2, 3, 5, 0, 0], dtype=torch.long)
features = torch.tensor([
    [ 1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.],
    [ 9., 10., 11., 12., 13., 14., 15., 16.],
    [17., 18., 19., 20., 21., 22., 23., 24.],
    [25., 26., 27., 28., 29., 30., 31., 32.]])

output[:, indices] = features
print(output)

'''
tensor([[ 8.,  1.,  4.,  5.,  0.,  6.,  0.,  0.,  0.],
        [16.,  9., 12., 13.,  0., 14.,  0.,  0.,  0.],
        [24., 17., 20., 21.,  0., 22.,  0.,  0.,  0.],
        [32., 25., 28., 29.,  0., 30.,  0.,  0.,  0.]])

Step	   features 的第几列	 对应 indices 的值	      output 的位置
1	          features[:, 0]	1	        -->        写入 output[:, 1]
2	          features[:, 1]	0	        -->        写入 output[:, 0]
3	          features[:, 2]	0	        -->        写入 output[:, 0] (覆盖前一步)
4	          features[:, 3]	2	        -->        写入 output[:, 2]
5	          features[:, 4]	3	        -->        写入 output[:, 3]
6	          features[:, 5]	5	        -->        写入 output[:, 5]
7	          features[:, 6]	0	        -->        写入 output[:, 0] (再次覆盖)
8	          features[:, 7]	0	        -->        写入 output[:, 0] (最终覆盖)

当indices 中出现重复坐标时， 后来者覆盖  
观察 indices, 数字 0 出现了 4 次：
indices = [1, 0, 0, 2, 3, 5, 0, 0]

对于 output 的第 0 列（ output[:, 0] ），它被赋值了四次，分别来自 features 的第 1, 2, 6, 7 列。
结果：在赋值操作完成后， output[:, 0] 的值将等于 features 中最后一次出现该索引的数据，即 features[:, 7]。
最终 output[:, 0] 的内容：
[8., 16., 24., 32.] （即 features 的最后一列）
'''

print("-----------------------------------------")

indices = torch.tensor([1, 4, 6, 2, 3, 5, 8, 7], dtype=torch.long)
features = torch.tensor([
    [ 1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.],
    [ 9., 10., 11., 12., 13., 14., 15., 16.],
    [17., 18., 19., 20., 21., 22., 23., 24.],
    [25., 26., 27., 28., 29., 30., 31., 32.]])
output = torch.zeros(4, 9, dtype=torch.float32)

output[:, indices] = features
print(output)

'''
Step	   features 的第几列	 对应 indices 的值	      output 的位置
1	          features[:, 0]	1	        -->        写入 output[:, 1]
2	          features[:, 1]	4	        -->        写入 output[:, 4]
3	          features[:, 2]	6	        -->        写入 output[:, 6]
4	          features[:, 3]	2	        -->        写入 output[:, 2]
5	          features[:, 4]	3	        -->        写入 output[:, 3]
6	          features[:, 5]	5	        -->        写入 output[:, 5]
7	          features[:, 6]	8	        -->        写入 output[:, 8]
8	          features[:, 7]	7	        -->        写入 output[:, 7]

tensor([[ 0.,  1.,  4.,  5.,  2.,  6.,  3.,  8.,  7.],
        [ 0.,  9., 12., 13., 10., 14., 11., 16., 15.],
        [ 0., 17., 20., 21., 18., 22., 19., 24., 23.],
        [ 0., 25., 28., 29., 26., 30., 27., 32., 31.]])
'''

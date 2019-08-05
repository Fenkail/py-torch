# 2019年07月31日13:59:02
# pytorch 第一步

import torch
import time

# 一个Tensor 只能为一个数 t小写
tensor = torch.tensor(3)
# Tensor 可以先声明空间
tensor2 = torch.Tensor(3,4)


# 声明张量，只分配了空间，未初始化
x = torch.Tensor(5,3)
print(x)
# 0-1 均匀分布随机初始化
y = torch.rand(5,3)
z = torch.rand(5,3)
print(y)
# 加法
print(y+z)
# 下面这种加法，把数据加到z上面去了
print(torch.add(y,z))
# numpy ==> tensor  两者共享内存 转换很快
a = torch.ones(5)
b = a.numpy()
c = torch.from_numpy(b)
# cup 与 GPU时间对比  由于需要将数据搬运到显存，所以GPU并非快
# cup ： 5.245208740234375e-06
# GPU ： 0.027654170989990234
start = time.time()
aa = y + z
end = time.time()
print(end-start)
y = y.cuda()
z = z.cuda()
start = time.time()
aa = y + z
end = time.time()
print(end-start)




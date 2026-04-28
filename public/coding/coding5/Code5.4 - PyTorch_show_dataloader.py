import numpy as np
import torch
from matplotlib.pyplot import *

#https://github.com/deep-learning-with-pytorch/dlwpt-code/blob/master/p1ch5/2_autograd.ipynb

#%%Part 1. Dataset
import os
import torch
from torchvision.io import read_image
from torch.utils.data import DataLoader,TensorDataset

torch.manual_seed(1)    # reproducible

BATCH_SIZE = 5      # 批训练的数据个数

x = torch.linspace(1, 10, 10)       # x data (torch tensor)
y = torch.linspace(10, 1, 10)       # y data (torch tensor)

# 先转换成 torch 能识别的 Dataset
torch_dataset = TensorDataset(x,y)

# 把 dataset 放入 DataLoader
loader = DataLoader(
    dataset=torch_dataset,      # torch TensorDataset format
    batch_size=BATCH_SIZE,      # mini batch size
    shuffle=True,               # 要不要打乱数据 (打乱比较好)
    num_workers=2,              # 多线程来读数据,win = 0, Linux根据实际配置设置
)
#train_loader = DataLoader(train_data, batch_size=batch_size, num_workers=4, shuffle=True, drop_last=True)
#val_loader   = DataLoader(val_data, batch_size=batch_size, num_workers=4, shuffle=False)

x_batch, y_batch = next(iter(loader)) 
print(x_batch,y_batch)



for epoch in range(3):   # 训练所有!整套!数据 
    for step, (batch_x, batch_y) in enumerate(loader):  # 每一步 loader 释放一小批数据用来学习
        # 假设这里就是你训练的地方...
        
        # 打出来一些数据
        print('Epoch: ', epoch, '| Step: ', step, '| batch x: ',
              batch_x.numpy(), '| batch y: ', batch_y.numpy())
        
        
#%%Part 2: Network
import torch
from torch import nn

##1.
class MLP(nn.Module):
  # 声明带有模型参数的层，这里声明了两个全连接层
  def __init__(self, **kwargs):
    # 调用MLP父类Block的构造函数来进行必要的初始化。这样在构造实例时还可以指定其他函数
    super(MLP, self).__init__(**kwargs)
    self.hidden = nn.Linear(32, 64)
    self.act = nn.ReLU()
    self.output = nn.Linear(64,10)
    
   # 定义模型的前向计算，即如何根据输入x计算返回所需要的模型输出
  def forward(self, x):
    o = self.act(self.hidden(x))
    return self.output(o)  

net=MLP()
print(net)
x = torch.rand(1,32)
yhat = net(x)

##2.直接采用nn.sequential定义
net=nn.Sequential( 
    nn.Linear(32,64),
    nn.ReLU(inplace=True),
    nn.Linear(64,1))

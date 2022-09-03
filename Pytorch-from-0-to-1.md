# Pytorch-from-0-to-1

[TOC]

## 第一章 导学

### 前置知识

- python编程基础
- 线性代数/概率论等数学知识
- Linux编程经验
- 机器学习基本概念

### pytorch简介

#### pytorch是什么？

> pytorch是一种开源的深度学习框架，与tensorflow, paddle一样可用于解决深度学习任务

- 个人学习经历：paddle的学习可以查看AI studio网站

#### 为什么要选择Pytorch?

- 逐渐形成完整的开发生态，资源丰富
- 代码简洁，易于理解，易于调试
- 动态图架构，运行速度快

### 课程目标

- 熟练掌握Pytorch的框架基础和API
- 熟练搭建卷积神经网络
- 掌握调参技巧，理解ML思想

### 课程核心内容

- 框架与基础
- 神经网络
- CV实战
- 模型保存与部属

## 第二章 环境搭建

### 环境配置

1.安装Anaconda, Pycharm

2.安装Nvidia Drive

https://www.nvidia.com/Download/index.aspx?lang-cn

参考：https://zhuanlan.zhihu.com/p/130524345

3.安装CUDA和cudnn

4.pytorch安装

```md
# conda换源
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
conda config --set show_channel_urls yes
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/
```

https://pytorch.org/get-started/locally/

```md
# 创建虚拟环境并安装
conda create -n yolov5 python=3.8
conda activate yolov5
conda install pytorch torchvision torchaudio cudatoolkit=9.2
```

检验是否安装成功：

```python
import torch
print(torch.cuda.is_available()) # True即为成功
```

## 第三章 Pytorch入门基础

### 机器学习中的分类和回归任务

- 分类-图像识别，目标检测
- 回归-目标检测

### 机器学习问题的构成元素

- 样本-数据X和标签y
- 模型
- 训练-学习的过程，通过数据拟合参数
- 测试-评价模型的方法
- 推理-预测标签的过程

### Pytorch中的基本概念

#### Tensor的基本定义

标量，向量，矩阵都是张量，标量是零维的张量，向量是一维的张量，矩阵是二维的张量。

Tensor编程实例

```python
import torch

a = torch.Tensor([[1, 2], [3, 4]])
print(a)
print(a.type())

# 创建2行3列的零矩阵
a = torch.Tensor(2, 3) 
print(a)
print(a.type())
# ones
a = torch.ones(2, 2)
print(a)
print(a.type())
# 对角线为1
a = torch.eye(2, 2)
print(a)
print(a.type())
# zeros
a = torch.zeros(2, 2)
print(a)
print(a.type())
# zero_like
b = torch.Tensor(2, 3)
b = torch.zero_like(b)
print(b)
print(b.type())
# ones_like
b = torch.Tensor(2, 3)
b = torch.ones_like(b)
print(b)
print(b.type())
# 随机 
a = torch.rand(2, 2)
print(a)
print(a.type())
# 正态分布
a = torch.normal(mean=0.0, std=torch.rand(5))
print(a)
print(a.type())
#均匀分布
a = torch.Tensor(2, 2).uniform_(-1, 1)
print(a)
print(a.type())
# 序列
a = torch.range(0, 10, 1)
print(a)
print(a.type())
# 拿到等间隔的n个数字
a = torch.linspace(2, 10, 3)
print(a)
print(a.type())
# 打乱序列
a = torch.randperm(10)
print(a)
print(a.type())

import numpy as np
a = np.array([[1, 2], [2, 3]])
print(a)
# tensor与np.array的转换
```

#### Tensor的属性

每个Tensor有torch.dtype, torch.device, torch.layout三种属性

##### torch.device

torch.device标识了torch.Tensor对象在创建之后所存储在的设备名称。

torch.tensor([1,2,3], dtype=torch.float32, device=torch.device('cuda:0'))

##### 稀疏的张量

减少内存开销

torch.sparse_coo_tensor

#### Tensor的算术运算

##### 四则运算

- 加减乘除
- 矩阵运算

##### 其他运算

- 幂运算
- 对数运算

```python
import torch
a = torch.rand(2, 3)
b = torch.rand(2, 3)
print(a)
print(b)

# add
print(a+b)
print(a.add(b))
print(torch.add(a, b))
print(a)
print(a.add_(b))
print(a)

# sub
print(a-b)
print(a.sub(b))
print(torch.sub(a, b))
print(a)
print(a.sub_(b))
print(a)

# *
print(a*b)
print(a.mul(b))
print(torch.mul(a, b))
print(a)
print(a.mul_(b))
print(a)

# /
print(a/b)
print(a.div(b))
print(torch.div(a, b))
print(a)
print(a.div_(b))
print(a)

# 矩阵运算 matmul
a = torch.ones(1, 2)
b = torch.ones(2, 1)

print(a@b)
print(a.matmul(b))
print(torch.matmul(a, b))
print(torch.mm(a, b))
print(a.mm(b))

# 高维tensor 
a = torch.ones(1, 2, 3, 4)
b = torch.ones(1, 2, 4, 3)
print(a.matmul(b))

# pow
a = torch.tensor([1, 2])
print(torch.pow(2, 3))
print(a.pow(3))
print(a**3)
print(a.pow_(3))
print(a)

# exp
a = torch.tensor([1, 2])
print(torch.exp(a))
print(torch.exp_(a))
print(a.exp())
print(a.exp_())

# log
a = torch.tensor([1, 2], dtype=torch.float32)
print(torch.log(a))
print(torch.log_(a))
print(a.log())
print(a.log_())

# sqrt 
print(torch.sqrt(a))
print(a.sqrt())
```

#### Tensor的比较运算

- torch.eq()

#### torch.nn 库

nn库是专门为神经网络设计的模块化接口，其构建于autograd之上可用来定义和运行神经网络

- nn.Linear & nn.conv2d
- nn.Parameter
- nn.functional
- nn.Module
- nn.Sequential

#### torchvision库

torchvision是独立于pytorch的关于图像操作的工具库

- vision.datasets 
- vision.models 已经训练好的模型，如AlexNet, VGG
- vision.transforms 常用的图像操作，如随机切割，旋转
- vision.utils
- .....

## 第四章 搭建简单神经网络

### 神经网络的基本概念

- 输入层，隐藏层，输出层

- 神经元，感知器，激活函数

- 前向计算，反向传播

- 过拟合，欠拟合

- 正则化，模型的性能评价

### 搭建神经网络解决分类和回归问题

#### 01 波士顿房价预测任务

##### 模型训练脚本

```python
import torch
import torch.nn as nn
import numpy as np

# data
datafile = './housing.data'
data = np.fromfile(datafile, sep=' ')
feature_names = [ 'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE','DIS', 
                 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV' ]
feature_num = len(feature_names)
# 一维变成二维
data = data.reshape(data.shape[0]//feature_num, feature_num)

# print(data.shape)
# print(data[:6, :]) # 取出前六条数据

# 数据集构建
X = data[:, :-1]
y = data[:, -1]
# train_num = ratio*data.shape[0]
# 划分训练集
ratio = 0.8
train_num = ratio*data.shape[0]
X_train = X[:train_num :]
y_train = y[:train_num, :]
X_test = X[train_num:, :]
y_test = y[train_num:, :]

# net 
class Net(nn.Module):
    def __init__(self, n_feature, n_output):
        super(Net, self).__init__()
        self.hidden = nn.Linear(n_feature, 100) # 添加隐藏层
        self.predict = nn.Linear(100, n_output)

    def forward(self, x):
        out = self.hidden(x)
        out = torch.relu(out)
        out = self.predict(out)
        return out
    
net = Net(13, 1)

# loss
loss_func = nn.MSELoss()
# optimizer
# optimizer = torch.optim.SGD(net.parameters(), lr=0.0001)
optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)

# training 
for i in range(1000):
    # train
    x_data = torch.tensor(X_train, dtype=torch.float32)
    y_data = torch.tensor(y_train, dtype=torch.float32)
    pred = net.forward(x_data)
    pred = torch.squeeze(pred)
    loss = loss_func(pred, y_data)*0.001
    # print(pred.shape)
    # print(y_data.shape)

    optimizer.zero_grad() # 参数梯度置0
    loss.backward()
    optimizer.step()

    print("ite:{}, loss_train:{}".format(i, loss*1000))
    # 打印前十个预测值和实际标签值
    print(pred[0:10])
    print(y_data[0:10])

    # test
    x_data = torch.tensor(X_test, dtype=torch.float32)
    y_data = torch.tensor(y_test, dtype=torch.float32)
    pred = net.forward(x_data)
    pred = torch.squeeze(pred)
    loss_test = loss_func(pred, y_data)*0.001
    # print(pred.shape)
    # print(y_data.shape)

    optimizer.zero_grad() # 参数梯度置0
    loss_test.backward()
    optimizer.step()

    print("ite:{}, loss_test:{}".format(i, loss_test*1000))
    # 打印前十个预测值和实际标签值
    print(pred[0:10])
    print(y_data[0:10])

# 保存模型
torch.save(net, './model.pkl')
```



#### 02 手写数字识别任务

##### 模型训练脚本

```python
import torch
import torchvision.datasets as dataset
import torchvision.transforms as transforms
import torch.utils.data as data_utils
import torch.nn as nn
# data
train_data = dataset.MNIST(root='./mnist', train=True, 
                           transform=transforms.ToTensor(),
                           download=True)
test_data = dataset.MNIST(root='./mnist', train=False, 
                           transform=transforms.ToTensor(),
                           download=False)
# batch-size
train_loader = data_utils.Dataloader(dataset=train_data, batch_size=64,
                                     shuffle=True)
test_loader = data_utils.Dataloader(dataset=train_data, batch_size=64,
                                     shuffle=True)

# net
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv = nn.Sequential(
            # 灰度图1 
            nn.Conv2d(1, 32, kernel_szie=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.fc = nn.Linear(32*14*14, 10)

    def forward(self, x):
        out = self.conv(x)
        out = out.view(out.size()[0], -1)
        out = self.fc(out)
        return out

cnn = CNN()
cnn = cnn.cuda()
# loss
loss_func = nn.CrossEntropyLoss()
# optimizer
optimizer = torch.optim.Adam(cnn.parameters, lr=0.01)

# train
for epoch in range(10):
    for i, (images, labels) in enumerate(train_loader):
        images = images.cuda()
        labels = labels.cuda()

        outputs = cnn(images)
        loss = loss_func(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print("epoch is {}, ite is {}/{}, loss is {}".format(epoch+1, i, len(train_data)//64, loss.item()))

    # eval/test
    loss_test = 0
    accuracy = 0
    for i, (images, labels) in enumerate(test_loader):
        images = images.cuda()
        labels = labels.cuda()
        outputs = cnn(images)
        # [batch-size]
        # outputs = batchsize * cls_num
        loss_test += loss_func(outputs, labels)
        _, pred = outputs.max(1)
        accuracy += (pred == labels).sum().item()

    accuracy /= len(test_data)
    loss_test /= (len(test_data)//64)
    
    print("epoch is {}, accuary is {}, loss_test is {}".format(epoch+1, 
                                    accuracy, loss_test.item()))

# save
torch.save(cnn, "./model/mnist_model.pkl")
```




























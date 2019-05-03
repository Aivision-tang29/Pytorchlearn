import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

# --------------------准备数据集------------------
# Dataset, DataLoader
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), std =(0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)

testset = torchvision.datasets.CIFAR10(root='./data',train=False,
                                       transform=transform, download=True)

# 运行Pytorch tutorial代码报错：BrokenPipeError: [Errno 32] Broken pipe
# 源代码地址： Training a classifier (CIFAR10)
#
# 该问题的产生是由于windows下多线程的问题，和DataLoader类有关，具体细节点这里Fix memory leak when using multiple workers on Windows。
#
# 解决方案：
#
#     修改调用torch.utils.data.DataLoader()函数时的 num_workers 参数。该参数官方API解释如下：
#
# num_workers (int, optional) – how many subprocesses to use for data loading. 0
# means that the data will be loaded in the main process. (default: 0)
#     该参数是指在进行数据集加载时，启用的线程数目。截止当前2018年5月9日11:15:52，如官方未解决该BUG，则可以通过修改num_works参数为 0 ，只启用一个主进程加载数据集，避免在windows使用多线程即可。

trainloader = DataLoader(dataset=trainset, batch_size=4, shuffle=True, num_workers=0)
testloader = DataLoader(dataset=testset, batch_size=4, shuffle=True, num_workers=0)


#定义一个简单的网络
# LeNet -5
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.fc1 = nn.Linear(in_features=16 * 5 * 5,out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.fc3 = nn.Linear(in_features=84, out_features=10)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool1(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)              # reshape tensor
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net=Net()
crterion=nn.CrossEntropyLoss()
optimzer=optim.SGD(net.parameters(),lr=0.001,momentum=0.9)

net=torch.load('cifar10.pth')
# for epoch in range(20):
#     running_loss=0.0
#     for i,data in enumerate(trainloader,0):
#         inputs,labels=data
#         # 清零梯度
#         optimzer.zero_grad()
#
#         output=net(inputs)
#         # 计算loss
#         loss=crterion(output,labels)
#         # 梯度计算
#         loss.backward()
#         # 更新参数
#         optimzer.step()
#
#         running_loss+=loss.item()
#
#         if i%2000==1999:
#             print('[%d,%5d] loss:%.3f'%(epoch+1,i+1,running_loss/2000))
#             running_loss=0.0
#
# print('training finish!')
#
# torch.save(net,'cifar10.pth')

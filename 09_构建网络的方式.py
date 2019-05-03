import torch
import torch.nn.functional as F
import torch.nn as nn
# torch.nn.Module类的主要方法
# 方法一：
# add_module( name, module)
# 功能： 给当前的module添加一个字module
# 参数：
# name -------子module的名称
# module-------子module

class Net1(nn.Module):

    def __init__(self):
        super(Net1, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3,
                               out_channels=32,
                               kernel_size=3,
                               stride=1,
                               padding=1)
        # why 32*32*3
        self.fc1 = nn.Linear(in_features=32 * 3 * 3,
                             out_features=128)
        self.fc2 = nn.Linear(in_features=128,
                             out_features=10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

print("CNN model_1:")
model_1 = Net1()
print(model_1)

# 使用torch.nn.Sequential

class Net2(nn.Module):
    def __init__(self):
        super(Net2, self).__init__()
        self.conv1=nn.Sequential(
            nn.Conv2d(in_channels=3,
                      out_channels=12,
                      kernel_size=3,
                      stride=1,
                      padding=1
                      ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.fc1=nn.Sequential(
            nn.Linear(in_features=32*3*3,
                      out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128,
                      out_features=10)
        )
    def forward(self, x):
        conv_out=self.conv1(x)
        res=conv_out.view(conv_out.size(0),-1)
        out=self.fc(res)
        return out

print('CNN model_2:')
print(Net2())

'''
Sequential,可以指定名字
使用字典OrderedDict形式
'''
from collections import OrderedDict

class Net3(nn.Module):

    def __init__(self):
        super(Net3, self).__init__()
        self.conv = nn.Sequential(
            OrderedDict(
                [
                    ('conv1', nn.Conv2d(in_channels=3,
                                        out_channels=32,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)),
                    ('relu1', nn.ReLU()),
                    ('pool1', nn.MaxPool2d(kernel_size=2))

                ]
            )
        )

        self.fc = nn.Sequential(
            OrderedDict(
                [
                    ('fc1', nn.Linear(in_features=32 * 3 * 3,
                                      out_features=128)),

                    ('relu2', nn.ReLU()),

                    ('fc2', nn.Linear(in_features=128,
                                      out_features=10))
                ]
            )
        )

    def forward(self, x):
        conv_out = self.conv(x)
        res = conv_out.view(conv_out.size(0), -1)
        out = self.fc(res)
        return out


print('CNN model_3:')
print(Net3())

# 使用add_module方法

class Net4(nn.Module):
    def __init__(self):
        super(Net4, self).__init__()
        self.conv = nn.Sequential()
        self.conv.add_module(name='conv1',
                             module=nn.Conv2d(in_channels=3,
                                              out_channels=32,
                                              kernel_size=1,
                                              stride=1))
        self.conv.add_module(name='relu1', module=nn.ReLU())
        self.conv.add_module(name='pool1', module=nn.MaxPool2d(kernel_size=2))

        self.fc = nn.Sequential()
        self.fc.add_module('fc1', module=nn.Linear(in_features=32 * 3 * 3,
                                                   out_features=128))
        self.fc.add_module('relu2', module=nn.ReLU())
        self.fc.add_module('fc2', module=nn.Linear(in_features=128,
                                                   out_features=10))

    def forward(self, x):
        conv_out = self.conv(x)
        res = conv_out.view(conv_out.size(0), -1)
        out = self.fc(x)
        return out


print('CNN model_4:')
print(Net4())
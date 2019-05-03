import torch
import torch.nn as nn
import torch.nn.functional as F


class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1=nn.Conv2d(in_channels=1,out_channels=6,kernel_size=5)
        self.conv2=nn.Conv2d(in_channels=6,out_channels=16,kernel_size=5)
        self.avgpool=nn.AvgPool2d(kernel_size=2,stride=2)
        self.fc1=nn.Linear(16*5*5,120)
        self.fc2=nn.Linear(120,84)
        self.fc3=nn.Linear(84,10)
    def forward(self,x):
        out=F.max_pool2d(F.relu(self.conv1(x)),(2,2))
        out=self.avgpool(F.relu(self.conv2(out)))
        out=out.view(-1,16*5*5)
        out=F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out


net=LeNet5()
print(net)
# params=list(net.parameters())
# print(params)
input = torch.randn(1, 1, 32, 32)
out = net(input)
print(out)
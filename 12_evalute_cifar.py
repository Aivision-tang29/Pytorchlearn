import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

CFAIR10_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'forg', 'horse', 'ship', 'truck']

image=Image.open('dog.jpg')

transform = transforms.Compose(
    [transforms.Resize((32, 32)),
     transforms.ToTensor(),
     transforms.Normalize(
         mean=(0.5, 0.5, 0.5),
         std=(0.5, 0.5, 0.5)
     )])

image_transformed = transform(image)
print(image_transformed.size())


# net=torch.load('cifar10.pth')
#
# img_trans=image_transformed.unsequeeze(0)
# output=net(img_trans)
#
# pred,predidx=torch.max(output,1)
#
# plt.figure()
# plt.imshow(np.array(image))
# plt.title(CFAIR10_names[predidx])
# plt.axis('off')
# plt.show()

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


CFAIR10_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'forg', 'horse', 'ship', 'truck']

# --------------测试数据集------------------------------
transform = transforms.Compose(
    [transforms.Resize((32, 32)),
     transforms.ToTensor(),
     transforms.Normalize(
         mean=(0.5, 0.5, 0.5),
         std=(0.5, 0.5, 0.5)
     )])

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=0)

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



if __name__ == '__main__':
    # net = Net()
    net=torch.load('cifar10.pth')

    correct=0
    total=0
    count=0

    with torch.no_grad():
        for sample_batch in testloader:
            images=sample_batch[0]
            labels=sample_batch[1]

            out=net(images)

            _,pred=torch.max(out,1)

            correct+=(pred==labels).sum().item()
            total+=labels.size(0)
            print('batch:{}'.format(count+1))
            count+=1

    acc=float(correct)/total
    print("acc:",acc)

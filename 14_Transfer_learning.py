import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from  torch.utils.data import DataLoader
import torchvision
from torchvision.transforms import transforms as tfs
from torchvision import models
from torchvision.models import ResNet
import numpy as np
from visdom import Visdom
import os

data_dir='./data/hymenoptera_data'

train_dataset=torchvision.datasets.ImageFolder(root=os.path.join(data_dir,'train'),
                                               transform=tfs.Compose([
                                                tfs.RandomResizedCrop(224),
                                                tfs.RandomHorizontalFlip(),
                                                tfs.ToTensor(),
                                                tfs.Normalize(
                                                mean=(0.485, 0.456, 0.406),
                                                std=(0.229, 0.224, 0.225)
                                                )
                                               ])
                                               )
val_dataset=torchvision.datasets.ImageFolder(root=os.path.join(data_dir,'val'),
                                               transform=tfs.Compose([
                                                tfs.RandomResizedCrop(224),
                                                tfs.RandomHorizontalFlip(),
                                                tfs.ToTensor(),
                                                tfs.Normalize(
                                                mean=(0.485, 0.456, 0.406),
                                                std=(0.229, 0.224, 0.225)
                                                )
                                               ])
                                               )


train_loader=DataLoader(train_dataset,batch_size=4,shuffle=True)
val_loader=DataLoader(val_dataset,batch_size=4,shuffle=True)

classname=train_dataset.classes
print('class_names:{}'.format(classname))

device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device.type)

viz=Visdom(env='transfer')
x,y=0,0
win = viz.line(
    X=np.array([x]),
    Y=np.array([y]),
    opts=dict(title='loss'))


model=models.vgg11(pretrained=True)
print(model)
in_channels=model.classifier[6].in_features
model.fc=nn.Linear(in_channels,2)

model=model.to(device)

lossfunc=nn.CrossEntropyLoss()

optimzer=optim.SGD(model.parameters(),lr=0.0001,momentum=0.9)

exp_lr_scheduler=optim.lr_scheduler.StepLR(optimizer=optimzer,step_size=10,gamma=0.5)

num_epoch=20

for epoch in range(num_epoch):
    running_loss=0.0
    exp_lr_scheduler.step()
    for i,sample in enumerate(train_loader):
        imgs,labels=sample

        model.train()

        imgs=imgs.to(device)
        labels=labels.to(device)

        optimzer.zero_grad()

        output=model(imgs)

        loss=lossfunc(output,labels)
        loss.backward()
        optimzer.step()

        running_loss+=loss.item()
        if i%5==4:
            correct=0
            total=0
            model.eval()
            for img,label in val_loader:
                img=img.to(device)
                label=label.to(device)
                out=model(img)
                _,pred=torch.max(out,1)
                correct+=((pred==label).sum().item())
                total+=imgs.size(0)
            print('epoch:{:f}{:f},loss:{:.5f},accuracy:{:.5f}%'.format(i+1,epoch,running_loss/5,correct*100/total))
        viz.line(X=np.array([i]),Y=np.array([running_loss/5]),win=win,update='append')
        running_loss=0
print("training finish!")
torch.save(model.state_dict(),'./model/transfer_model.pth')


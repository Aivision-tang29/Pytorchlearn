import os

filename_list=os.listdir('./test_images')
print(filename_list)

with open('image_name.txt','a+') as f:
    for idx,fname in enumerate(filename_list):
        write_str=fname+'\t'+str(idx)+'\n'
        f.write(write_str)
    f.close()

import numpy as np
from skimage import io
from skimage import transform
import matplotlib.pyplot as plt
import os
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
from torchvision.utils import make_grid
from PIL import Image
from torchvision import datasets
# dataset 类是要复写 __len__ __getitem__
class Mydatasets1(Dataset):
    def __init__(self,root_dir,transform=None):
        self.size=len(os.listdir(root_dir))
        self.filename=[]
        self.filelabel = []
        self.transform=transform
        for idx,fname in enumerate(os.listdir(root_dir)):
            self.filename.append(fname)
        with open('image_name.txt','r') as f:
            for i in range(self.size):
                str=f.readline()
                label=str.split('\t')[1]
                self.filelabel.append(label)

        # 读取图像
        # 读取标签
        # label+4cordinate
    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        # return image,label
        image=Image.open('./test_images/'+filename_list[idx])
        label=self.filelabel[idx]
        image=self.transform(image)
        return image,label

tfs1=transforms.Compose([
                         transforms.CenterCrop(256),
                         transforms.ToTensor()
                         ])

dataset1=Mydatasets1('./test_images',transform=tfs1)
print(len(dataset1))

# for idx,data in enumerate(dataset1):
#     image,label=data[0],data[1]
#     print(image,label)


# dataloader 加载数据时候的minibatch，shuffle

data_loader=DataLoader(dataset1,batch_size=2,shuffle=True,num_workers=2)

# 另外的一种更加简单的  ImageFolder

dataset2=datasets.ImageFolder(root='./test_images')

for idx,data in enumerate(dataset2):
     image,label=data[0],data[1]
     print(image,label)
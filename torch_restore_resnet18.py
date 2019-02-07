import torch
import torchvision
from torchvision import datasets,transforms
import os
import matplotlib.pyplot as plt
import time
from torchvision import models
from torch.autograd import Variable

data_dir="DogsvsCats"
data_tranform={x:transforms.Compose(
    [
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
    ]
) for x in ["train","val","test"]}

image_datasets={x:datasets.ImageFolder(root=os.path.join(data_dir,x),
                                       transform=data_tranform[x])
                for x in ["train","val","test"]}

model=models.resnet18(pretrained=False)
model.fc=torch.nn.Sequential(torch.nn.Linear(512,2))
model.load_state_dict(torch.load('model_resnet18_finetune.pkl'))
print(model)

data_test_img = datasets.ImageFolder(root=os.path.join(data_dir,"test"),
                                     transform = data_tranform["test"])
data_loader_test_img = torch.utils.data.DataLoader(dataset=data_test_img,
                                                   batch_size = 64)

image, label = next(iter(data_loader_test_img))
images = Variable(image)
y_pred = model(images)
_, pred = torch.max(y_pred.data, 1)
print(pred)
classes =image_datasets["train"].classes
img = torchvision.utils.make_grid(image)
img = img.numpy().transpose(1,2,0)
mean = [0.5,0.5,0.5]
std  = [0.5,0.5,0.5]
img = img*std+mean
print("Pred Label:",[classes[i] for i in pred])
plt.imshow(img)
plt.show()
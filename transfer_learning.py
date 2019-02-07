import torch
import torchvision
from torchvision import datasets,transforms
import os
import matplotlib.pyplot as plt
import time
from torchvision import models
from torch.autograd import Variable
data_dir="DogsvsCats"
# print(os.path.join(data_dir,"train1"))
data_tranform={x:transforms.Compose(
    [
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
    ]
) for x in ["train","val"]}

image_datasets={x:datasets.ImageFolder(root=os.path.join(data_dir,x),
                                       transform=data_tranform[x])
                for x in ["train","val"]}

data_loader={x:torch.utils.data.DataLoader(dataset=image_datasets[x],
                                           batch_size=16,
                                           shuffle=True)
             for x in ["train","val"]}

X_train,y_train=next(iter(data_loader["train"]))
print(len(X_train))
print(len(y_train))

mean = [0.5,0.5,0.5]
std  = [0.5,0.5,0.5]
classes =image_datasets["train"].classes
print(classes)

img=torchvision.utils.make_grid(X_train)
img=img.numpy().transpose([1,2,0])
img=img*std+mean
print([classes[i] for i in y_train])
plt.imshow(img)
plt.show()

model=models.resnet18(pretrained=True)
print(model)

for parma in model.parameters():
    parma.requires_grad = False

model.fc=torch.nn.Sequential(torch.nn.Linear(512,2))

cost = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.fc.parameters())
print(model)

n_epochs = 1
use_gpu=False
for epoch in range(n_epochs):
    since = time.time()
    print("Epoch{}/{}".format(epoch, n_epochs))
    print("-" * 10)
    for param in ["train", "val"]:
        if param == "train":
            model.train = True
        else:
            model.train = False

        running_loss = 0.0
        running_correct = 0
        batch = 0
        for data in data_loader[param]:
            batch += 1
            X, y = data
            if use_gpu:
                X, y = Variable(X.cuda()), Variable(y.cuda())
            else:
                X, y = Variable(X), Variable(y)

            optimizer.zero_grad()
            y_pred = model(X)
            _, pred = torch.max(y_pred.data, 1)

            loss = cost(y_pred, y)
            if param == "train":
                loss.backward()
                optimizer.step()
            running_loss += loss.data
            running_correct += torch.sum(pred == y.data)
            if batch % 100 == 0 and param == "train":
                print("Batch {}, Train Loss:{:.4f}, Train ACC:{:.4f}".format(
                    batch, running_loss / (16 * batch), 100 * running_correct / (16 * batch)))

        epoch_loss = running_loss / len(image_datasets[param])
        epoch_correct = 100 * running_correct / len(image_datasets[param])

        print("{}  Loss:{:.4f},  Correct{:.4f}".format(param, epoch_loss, epoch_correct))
    now_time = time.time() - since
    print("Training time is:{:.0f}m {:.0f}s".format(now_time // 60, now_time % 60))



torch.save(model.state_dict(), "model_resnet18_finetune.pkl")

data_test_img = datasets.ImageFolder(root="test1",
                                     transform = data_tranform)
data_loader_test_img = torch.utils.data.DataLoader(dataset=data_test_img,
                                                   batch_size = 16)

image, label = next(iter(data_loader_test_img))
images = Variable(image.cuda())
y_pred = model(images)
_, pred = torch.max(y_pred.data, 1)
print(pred)

img = torchvision.utils.make_grid(image)
img = img.numpy().transpose(1,2,0)
mean = [0.5,0.5,0.5]
std  = [0.5,0.5,0.5]
img = img*std+mean
print("Pred Label:",[classes[i] for i in pred])
plt.imshow(img)
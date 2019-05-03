import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision.models import AlexNet
import matplotlib.pyplot as plt


model = AlexNet(num_classes=2)
optimizer = optim.SGD(params=model.parameters(), lr=0.05)

scheduler=lr_scheduler.StepLR(optimizer,step_size=30,gamma=0.1)

plt.figure()

x=list(range(100))
y=[]
for epoch in range(100):
    scheduler.step()
    print(epoch,scheduler.get_lr()[0])
    y.append(scheduler.get_lr()[0])

plt.plot(x,y)
plt.show()
print()
plt.figure()
y.clear()
scheduler = lr_scheduler.MultiStepLR(optimizer, [30, 80], 0.1)
for epoch in range(100):
    scheduler.step()
    print(epoch, 'lr={:.6f}'.format(scheduler.get_lr()[0]))
    y.append(scheduler.get_lr()[0])

plt.plot(x, y)
plt.show()

scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
print()
plt.figure()
y.clear()
for epoch in range(100):
    scheduler.step()
    print(epoch, 'lr={:.6f}'.format(scheduler.get_lr()[0]))
    y.append(scheduler.get_lr()[0])

plt.plot(x, y)
plt.show()

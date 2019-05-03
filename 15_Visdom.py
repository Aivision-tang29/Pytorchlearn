from visdom import Visdom
import numpy as np
vis=Visdom(env='test')

x=np.linspace(0,2*np.pi,num=180)
y=np.sin(x)

vis.line(Y=y,X=x)

y2=np.cos(x)
vis.line(Y=np.column_stack((y,y2)))
help(vis.line)

loss_win = vis.line(np.arange(10))
# loss_win = vis.line(np.arange(10))

# 向散点图中加入新的描述
vis.scatter(
    X=np.random.rand(255,2),
    win=loss_win
)
x=np.linspace(0,100,10)
y=np.sin(x)
acc_cure=vis.line(X=x,Y=y)
import time
for i in range(100):
    vis.line(X=x,Y=np.sin(x),win=acc_cure,update='append')
    time.sleep(1)
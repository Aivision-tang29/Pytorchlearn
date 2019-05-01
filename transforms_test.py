import torch
import torchvision.transforms as tfs
from PIL import Image


image=Image.open('./test_images/1.jpeg')
# image.show()

# 数据类型的转化
tfs1=tfs.Compose([tfs.ToTensor()])

img1=tfs1(image)
print(img1)


# 标准化 Normalize(必须是对tensor处理)
tfs2=tfs.Compose([tfs.Normalize(mean=(0.5,0.5,0.5),std=(0.5,0.5,0.5))])
img2=tfs2(img1)
print(img2)

# 这里不需要对tensor处理
tfs3=tfs.Compose([tfs.Resize(256)])
print(img1.shape)
img3=tfs3(image)
# img3.show()

# 中心裁剪
tfs4=tfs.Compose([tfs.CenterCrop(128)])
img4=tfs4(image)
# img4.show()


# 随机裁剪 固定大小 512
tfs5=tfs.Compose([tfs.RandomCrop(512,padding=0)])
img5=tfs5(image)
# img5.show()


#转灰度图
tfs6=tfs.Compose([tfs.Grayscale(num_output_channels=1)])
img6=tfs6(image)
# img6.show()


transforms7 = tfs.Compose([tfs.ColorJitter()])
img7 = transforms7(image)
# img7.show()



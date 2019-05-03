import matplotlib.pyplot as plt
import skimage.io as io
import cv2
from PIL import Image
import numpy as np
import torch


# np.ndarray (H,W,C) RGB
img_skimage=io.imread('./test_images/9.jpeg')
print(img_skimage.shape)
print(img_skimage.dtype)
# np.array (H,W,C) BGR
img_opencv=cv2.imread('./test_images/9.jpeg')
print(img_opencv.shape)

img_pil=Image.open('./test_images/9.jpeg')
#PIL.Image.Image 对象
img_pil_nparray=np.array(img_pil)
print(img_pil_nparray.shape)


plt.figure()

for i,im in enumerate([img_skimage,img_opencv,img_pil_nparray]):
    ax=plt.subplot(1,3,i+1)
    ax.imshow(im)
    plt.pause(0.01)
plt.show()


# 图像转 torch tensor
tensor_skimage=torch.from_numpy(np.transpose(img_skimage,(2,0,1)))
tensor_cv=torch.from_numpy(np.transpose(img_opencv,(2,0,1)))
tensor_pil=torch.from_numpy(np.transpose(img_pil_nparray,(2,0,1)))

img_skimage_2=np.transpose(tensor_skimage.numpy(),(1,2,0))
img_cv_2 = np.transpose(tensor_cv.numpy(), (1, 2, 0))
img_pil_2 = np.transpose(tensor_pil.numpy(), (1, 2, 0))

plt.figure()
for i, im in enumerate([img_skimage_2, img_cv_2, img_pil_2]):
    ax = plt.subplot(1, 3, i + 1)
    ax.imshow(im)
    plt.pause(0.01)

plt.show()


# opencv BGR--> RGB

img_cv=cv2.cvtColor(img_cv_2,cv2.COLOR_BGR2RGB)

tensor_cv=torch.from_numpy(img_cv)

img_cv_2=tensor_cv.numpy()

plt.figure()
plt.title('opencv image')
plt.imshow(img_cv_2)
plt.show()
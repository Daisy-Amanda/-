#%%
import cv2
import numpy as np
import imutils
import matplotlib.pyplot as plt

def cv_show(name,img):#展示图片
    cv2.namedWindow(name,0)
    cv2.imshow(name,img)
    cv2.waitKey(0)
#读入图片
img=cv2.imread('C:\Users\HUAWEI\Desktop\微信图片_20240604091219.jpg')
#转换为灰度图
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#高斯滤波
blurred_gauss=cv2.GaussianBlur(gray,(3,3),0)
#增强亮度
def imgBrightness(img1,c,b):
    rows,cols=img1.shape
    blank=np.zeros([rows,cols],img1.dtype)
   
blurred_bright=imgBrightness(blurred_gauss,1.5,3)
#自适应二值化
blurred_threshold=cv2.adaptiveThreshold(blurred_bright,
                                        255,
                                        cv2.ADAPTIVE_THRESH_MEAN_C,
                                        cv2.THRESH_BINARY,51,2)
#显示原来的和缩放后的图像
fig=plt.figure(figsize=(16,12))

#Subplot for original image
a=fig.add_subplot(2,3,1)
imgplot=plt.imshow(img)
a.set_title('原始图片')

a=fig.add_subplot(2,3,2)
imgplot=plt.imshow(gray,cmap='gray')
a.set_title('灰度图')

a=fig.add_subplot(2,3,3)
imgplot=plt.imshow(blurred_gauss,cmap='gray')
a.set_title('高斯滤波')

a=fig.add_subplot(2,3,4)
imgplot=plt.imshow(blurred_bright,cmap='gray')
a.set_title('增强亮度')

a=fig.add_subplot(2,3,5)
imgplot=plt.imshow(blurred_threshold,cmap='gray')
a.set_title('自适应二值化')

plt.show()
#四点变换，划出选择题区域
paper=four_point_transform(img,np.arry(docCnt[0]).reshape(4,2))
warped=four_point_transform(gray,np.array(docCnt[0]).reshape(4,2))
#四点变换，划出准考证区域
ID_Area=four_point_transform(img,np.array(docCnt[1]).reshape(4,2))
ID_Area_warped=four_point_transform(gray,np.array(docCnt[2]).reshape(4,2))

#输出四点透视变换分割的区域
fig=plt.figure(figsize=(16,12))

#subplot for origiinal image
a=fig.add_subplot(2,3,1)
imgplot=plt.imshow(paper)
a.set_title('选择题区域')

a=fig.add_subplot(2,3,2)
imgplot=plt.imshow(ID_Area)
a.set_title('准考证区域')
# %%
#!/usr/bin/env python
# coding: utf-8

# In[31]:
from matplotlib import pyplot as plt
from skimage.util import random_noise
import numpy as np
import cv2




# In[32]:

count=6
maximgsinrow = 5


im = cv2.imread('Lena.png')
rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

#converting to grayscale
gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
# cv2.imshow('Grayscale Image', gray)

'''
adding guassian noise in set of 5,10,15,20,25,30

'''

#adding guassian noise in set of 5 (take 5 original image and apply guassian noise)

plt.figure(figsize=(20,20))
#while(count<=1):
g5_img = random_noise(gray,mode='gaussian',mean=0,var=0.01)
g5_img = np.array(255*g5_img, dtype = 'uint8') 

plt.subplot(1 + count, maximgsinrow, 1)
plt.imshow(g5_img,cmap = "gray")
#cv2.imshow('g5_img',g5_img) 

g5_img_mean = g5_img.copy()
g5_img_mean = g5_img_mean.astype("float")
# print("g5_img", g5_img)
# print("g5_img_mean", g5_img_mean)

for i in range(1,count*5,1):
    g5_img = random_noise(gray,mode='gaussian',mean=0,var=0.01)
    g5_img = np.array(255*g5_img, dtype = 'uint8') 
    plt.subplot(1 + count, maximgsinrow, i+1)
    plt.imshow(g5_img,cmap = "gray")
    g5_img_mean = np.add(g5_img_mean,g5_img)


g5_img_mean = np.divide(g5_img_mean,count*5)
g5_img_mean = g5_img_mean.astype('uint8')

#     cv2.imshow('blur for set {}'.format(count*5), g5_img_mean) 

# # g5_img_mean = g5_img_mean.astype('uint8')

# print(g5_img_mean)
# # cv2.imshow('blur',g5_img_mean) 

# # g5_img_mean = g5_img_mean/5

plt.subplot(1 + count, maximgsinrow, count*maximgsinrow + 1)
plt.title("set {}".format(count*5))
plt.imshow(g5_img_mean, cmap = "gray")

#count=count+1

    
plt.show()
    
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# In[33]:


# import cv2
# from matplotlib import pyplot as plt
# from skimage.util import random_noise
# import numpy as np


im = cv2.imread('Lena.png')
rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

#converting to grayscale
gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
# cv2.imshow('Grayscale Image', gray)

'''
adding guassian noise in set of 5,10,15,20,25,30

'''

#adding guassian noise in set of 5 (take 5 original image and apply guassian noise)

plt.figure(figsize=(20,20))
#while(count<=1):
g5_img = random_noise(gray, mode='s&p',amount=0.1)
g5_img = np.array(255*g5_img, dtype = 'uint8') 

plt.subplot(1 + count, maximgsinrow, 1)
plt.imshow(g5_img,cmap = "gray")
#cv2.imshow('g5_img',g5_img) 

g5_img_mean = g5_img.copy()
g5_img_mean = g5_img_mean.astype("float")
# print("g5_img", g5_img)
# print("g5_img_mean", g5_img_mean)

for i in range(1,count*5,1):
    g5_img = random_noise(gray, mode='s&p',amount=0.1)
    g5_img = np.array(255*g5_img, dtype = 'uint8') 
    plt.subplot(1 + count, maximgsinrow, i+1)
    plt.imshow(g5_img,cmap = "gray")
    g5_img_mean = np.add(g5_img_mean,g5_img)


g5_img_mean = np.divide(g5_img_mean,count*5)
g5_img_mean = g5_img_mean.astype('uint8')

#     cv2.imshow('blur for set {}'.format(count*5), g5_img_mean) 

# # g5_img_mean = g5_img_mean.astype('uint8')

# print(g5_img_mean)
# # cv2.imshow('blur',g5_img_mean) 

# # g5_img_mean = g5_img_mean/5

plt.subplot(1 + count, maximgsinrow, count*maximgsinrow + 1)
plt.title("set {}".format(count*5))
plt.imshow(g5_img_mean, cmap = "gray")

#count=count+1

    
plt.show()   
    
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# In[34]:


# import cv2
# from matplotlib import pyplot as plt
# from skimage.util import random_noise
# import numpy as np


im = cv2.imread('Lena.png')
rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

#converting to grayscale
gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
# cv2.imshow('Grayscale Image', gray)

'''
adding guassian noise in set of 5,10,15,20,25,30

'''

#adding guassian noise in set of 5 (take 5 original image and apply guassian noise)

plt.figure(figsize=(20,20))
#while(count<=1):
g5_img = random_noise(gray,mode='speckle',mean=0,var=0.01)
g5_img = np.array(255*g5_img, dtype = 'uint8') 

plt.subplot(1 + count, maximgsinrow, 1)
plt.imshow(g5_img,cmap = "gray")
#cv2.imshow('g5_img',g5_img) 

g5_img_mean = g5_img.copy()
g5_img_mean = g5_img_mean.astype("float")
# print("g5_img", g5_img)
# print("g5_img_mean", g5_img_mean)

for i in range(1,count*5,1):
    g5_img = random_noise(gray,mode='speckle',mean=0,var=0.01)
    g5_img = np.array(255*g5_img, dtype = 'uint8') 
    plt.subplot(1 + count, maximgsinrow, i+1)
    plt.imshow(g5_img,cmap = "gray")
    g5_img_mean = np.add(g5_img_mean,g5_img)


g5_img_mean = np.divide(g5_img_mean,count*5)
g5_img_mean = g5_img_mean.astype('uint8')

#     cv2.imshow('blur for set {}'.format(count*5), g5_img_mean) 

# # g5_img_mean = g5_img_mean.astype('uint8')

# print(g5_img_mean)
# # cv2.imshow('blur',g5_img_mean) 

# # g5_img_mean = g5_img_mean/5

plt.subplot(1 + count, maximgsinrow, count*maximgsinrow + 1)
plt.title("set {}".format(count*5))
plt.imshow(g5_img_mean, cmap = "gray")

#count=count+1

    
plt.show()   
    
# cv2.waitKey(0)
# cv2.destroyAllWindows()


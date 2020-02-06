'''
1. Download Lena image and scale it by factors of 1,2,0.5 using bilinear interpolation and display the scaled images. 
Also, display the output of built-in functions for doing scaling by factors of 0.5,2. 
Compare the results.
'''
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
from tqdm import tqdm

# 1
def BillinearInterpolation(Image, Scale=(2, 2)):
    NewSize = (int(round(Image.shape[0]*Scale[0])), int(round(Image.shape[1]*Scale[1])))
    ScaledImg = np.ones(NewSize) * -1

    for i in range(Image.shape[0]):
        for j in range(Image.shape[1]):
            if int(i % (1/Scale[0])) == 0 and int(j % Scale[1]) == 0:
                ScaledImg[int(i*Scale[0]), int(j*Scale[1])] = Image[i, j]

    # Fill Missing Spots
    for i in tqdm(range(NewSize[0])):
        for j in range(NewSize[1]):
            if ScaledImg[i, j] == -1:
                InvCo = (i/Scale[0], j/Scale[1])
                A1 = (InvCo[0] - int(InvCo[0]))*(InvCo[1] - int(InvCo[1]))
                A2 = (1 - (InvCo[0] - int(InvCo[0])))*(InvCo[1] - int(InvCo[1]))
                A3 = (InvCo[0] - int(InvCo[0]))*(1 - (InvCo[1] - int(InvCo[1])))
                A4 = (1 - (InvCo[0] - int(InvCo[0])))*(1 - (InvCo[1] - int(InvCo[1])))
                A = A4
                if int(InvCo[0]) + 1 < Image.shape[0] and int(InvCo[1]) + 1 < Image.shape[1]:
                    A += A1 + A2 + A3
                elif int(InvCo[0]) + 1 < Image.shape[0]:
                    A += A3
                elif int(InvCo[1]) + 1 < Image.shape[1]:
                    A += A2
                

                ScaledImg[i, j] = (A4*Image[int(InvCo[0]), int(InvCo[1])])/A
                if int(InvCo[0]) + 1 < Image.shape[0] and int(InvCo[1]) + 1 < Image.shape[1]:
                    ScaledImg[i, j] += (A1*Image[int(InvCo[0]) + 1, int(InvCo[1]) + 1])/A
                    ScaledImg[i, j] += (A3*Image[int(InvCo[0]) + 1, int(InvCo[1])])/A
                    ScaledImg[i, j] += (A2*Image[int(InvCo[0]), int(InvCo[1]) + 1])/A
                elif int(InvCo[0]) + 1 < Image.shape[0]:
                    ScaledImg[i, j] += (A3*Image[int(InvCo[0]) + 1, int(InvCo[1])])/A
                elif int(InvCo[1]) + 1 < Image.shape[1]:
                    ScaledImg[i, j] += (A2*Image[int(InvCo[0]), int(InvCo[1]) + 1])/A

    return ScaledImg

# Driver Code
inputImgPath = 'Assignment3/LenaImage.png'
I = cv2.imread(inputImgPath, 0)
Scale = (2, 2)
ScaledImg = BillinearInterpolation(I, Scale)
ScaledImg_cv2 = cv2.resize(I, (int(round(I.shape[0]*Scale[0])), int(round(I.shape[1]*Scale[1]))), interpolation=cv2.INTER_LINEAR)
ax = plt.subplot(1, 2, 1)
ax.title.set_text('CV2 Scaling')
plt.imshow(ScaledImg_cv2, 'gray')
ax = plt.subplot(1, 2, 2)
ax.title.set_text('Implementation')
plt.imshow(ScaledImg, 'gray')
plt.show()
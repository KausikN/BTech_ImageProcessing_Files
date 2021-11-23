'''
2. Download the leaning tower of PISA image and find the angle of inclination using appropriate rotations.
'''
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import imutils
import math


# 2
def BillinearInterpolation(Image, Angle=0.0):
    anglerad = -Angle * math.pi / 180.0
    NewSize = (Image.shape[0], Image.shape[1])
    RotImg = np.ones(NewSize) * -1

    for i in range(Image.shape[0]):
        for j in range(Image.shape[1]):
            newi = math.cos(anglerad)*i + math.sin(anglerad)*j
            newj = -1*math.sin(anglerad)*i + math.cos(anglerad)*j
            if (newi-float(int(newi))) == 0.0 and (newj-float(int(newj))) == 0.0:
                RotImg[int(newi), int(newj)] = Image[i, j]

    # Fill Missing Spots
    for i in tqdm(range(NewSize[0])):
        for j in range(NewSize[1]):
            if RotImg[i, j] == -1:
                InvCo = [math.cos(anglerad)*i - math.sin(anglerad)*j, math.sin(anglerad)*i + math.cos(anglerad)*j]
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
                
                if int(InvCo[0]) < Image.shape[0] and int(InvCo[1]) < Image.shape[1]:
                    RotImg[i, j] = (A4*Image[int(InvCo[0]), int(InvCo[1])])/A
                    if int(InvCo[0]) + 1 < Image.shape[0] and int(InvCo[1]) + 1 < Image.shape[1]:
                        RotImg[i, j] += (A1*Image[int(InvCo[0]) + 1, int(InvCo[1]) + 1])/A
                        RotImg[i, j] += (A3*Image[int(InvCo[0]) + 1, int(InvCo[1])])/A
                        RotImg[i, j] += (A2*Image[int(InvCo[0]), int(InvCo[1]) + 1])/A
                    elif int(InvCo[0]) + 1 < Image.shape[0]:
                        RotImg[i, j] += (A3*Image[int(InvCo[0]) + 1, int(InvCo[1])])/A
                    elif int(InvCo[1]) + 1 < Image.shape[1]:
                        RotImg[i, j] += (A2*Image[int(InvCo[0]), int(InvCo[1]) + 1])/A

    return RotImg

def RotateImage(Image, angle, cropBoundary=False):
    RotatedImage = None
    if cropBoundary:
        RotatedImage = imutils.rotate(Image, angle)
    else:
        RotatedImage = imutils.rotate_bound(Image, angle)
    return RotatedImage

# Driver Code
imgPath = 'Assignment3/LeaningTowerOfPisa.jpeg'
I = cv2.imread(imgPath, 0)

# Params
Angles = list(np.arange(6, 6 + 1, 1))
cropBoundary = True
maxColumns = 5
# Params

index = 1
nRows = int(len(Angles) / maxColumns) + int((len(Angles) / maxColumns) - int(len(Angles) / maxColumns))

for angle in tqdm(Angles):
    ax = plt.subplot(int(len(Angles) / maxColumns) + 1, maxColumns, index)
    ax.title.set_text(str(angle))
    #RotImg = RotateImage(I, angle, cropBoundary)
    RotImg = BillinearInterpolation(I, angle)
    plt.imshow(RotImg, 'gray')
    index += 1
plt.show()

# ANSWER IS AROUND 6 DEGREES
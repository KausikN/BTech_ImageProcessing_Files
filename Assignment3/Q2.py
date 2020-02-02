'''
2. Download the leaning tower of PISA image and find the angle of inclination using appropriate rotations with bilinear interpolation.
'''
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import imutils


# 2
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
Angles = list(np.arange(5, 6 + 1, 1))
cropBoundary = True
maxColumns = 5
# Params

index = 1
nRows = int(len(Angles) / maxColumns) + int((len(Angles) / maxColumns) - int(len(Angles) / maxColumns))

for angle in tqdm(Angles):
    plt.subplot(int(len(Angles) / maxColumns) + 1, maxColumns, index)
    plt.imshow(RotateImage(I, angle, cropBoundary), 'gray')
    index += 1
plt.show()

# ANSWER IS AROUND 6 DEGREES
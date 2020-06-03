'''
Codes for Morphological Processing
'''

# Imports
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Main Functions
def Erode(I, W, stride=(1, 1), binaryMatching=False):
    Img = I.copy()
    Window = W.copy()
    if I.ndim == 2:
        Img = np.reshape(Img, (Img.shape[0], Img.shape[1], 1))
        Window = np.reshape(Window, (Window.shape[0], Window.shape[1], 1))

    ErodedImg = np.zeros(Img.shape)

    WindowMidPoint = (int(Window.shape[0]/2), int(Window.shape[1]/2))
    print(WindowMidPoint)

    for i in tqdm(range(0, Img.shape[0], stride[0])):
        for j in range(0, Img.shape[1], stride[1]):
            matching = True
            for wi in range(Window.shape[0]):
                if (i + (wi-WindowMidPoint[0]) > Img.shape[0] - 1) or (i + (wi-WindowMidPoint[0]) < 0):
                    OutBounds_X = True
                for wj in range(Window.shape[1]):
                    if (j + (wj-WindowMidPoint[1]) > Img.shape[1] - 1) or (j + (wj-WindowMidPoint[1]) < 0):
                        OutBounds_Y = True
                    for c in range(Window.shape[2]):
                        if OutBounds_X or OutBounds_Y:
                            if not Window[wi, wj, c] == 0:
                                matching = False
                                break
                        elif not binaryMatching:
                            if not Window[wi, wj, c] == Img[i + (wi-WindowMidPoint[0]), j + (wj-WindowMidPoint[1]), c]:
                                matching = False
                                break
                        elif binaryMatching:
                            if Window[wi, wj, c] > 0 and Img[i + (wi-WindowMidPoint[0]), j + (wj-WindowMidPoint[1]), c] == 0:
                                matching = False
                                break
                            elif Window[wi, wj, c] == 0 and Img[i + (wi-WindowMidPoint[0]), j + (wj-WindowMidPoint[1]), c] > 0:
                                matching = False
                                break
                    if not matching:
                        break
                if not matching:
                    break
            if matching:
                ErodedImg[i, j, c] = 255

    if I.ndim == 2:
        ErodedImg = np.reshape(ErodedImg, (ErodedImg.shape[0], ErodedImg.shape[1]))
    
    return ErodedImg


# Driver Code
Img_Path = 'OtherCodes/Test.png'
WindowSize = (10, 10)
stride = (1, 1)
binaryMatching = True

Window = np.ones(WindowSize)
print(Window)
Img = cv2.imread(Img_Path, 0)
ax = plt.subplot(1, 2, 1)
ax.title.set_text('Original Img ' + str(Img.shape))
plt.imshow(Img)
# cv2.imshow("Original Img " + str(Img.shape), Img)
# cv2.waitKey(0)

ErodedImg = Erode(Img, Window, stride=stride, binaryMatching=binaryMatching)

ax = plt.subplot(1, 2, 2)
ax.title.set_text('Eroded Img ' + str(ErodedImg.shape))
plt.imshow(Img)
# cv2.imshow("Eroded Img", ErodedImg)
# cv2.waitKey(0)

plt.show()

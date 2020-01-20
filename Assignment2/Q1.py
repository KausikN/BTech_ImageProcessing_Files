'''
Take a Lena image and convert it into grayscale. 
Add three different types of noises(salt and pepper, additive Gaussian noise, speckle), 
each noise in the sets of 5,10,15,20,25,30. 
Take average for each set and display the average images. 
Report the observation made.
'''
import cv2
import numpy as np
import random

def rgb2gray(I):
    r, g, b = I[:,:,0], I[:,:,1], I[:,:,2]
    gray = (0.2989 * r + 0.5870 * g + 0.1140 * b)
    return gray

def SaltPepperNoise(I, prob):
    max = 255
    I_g = I
    probpercent = int(prob*100)
    # Greyscale
    if I_g.ndim == 2:
        for i in range(I_g.shape[0]):
            for j in range(I_g.shape[1]):
                r = random.randint(1, 100)
                if r <= probpercent:
                    if r <= int(probpercent / 2):
                        I_g[i, j] = max # Salt
                    else:
                        I_g[i, j] = 0 # Pepper
    # RGB
    elif I_g.ndim == 3:
        for i in range(I_g.shape[0]):
            for j in range(I_g.shape[1]):
                r = random.randint(1, 100)
                if r <= probpercent:
                    if r <= int(probpercent / 2):
                        I_g[i, j, 0] = max # Salt
                        I_g[i, j, 1] = max # Salt
                        I_g[i, j, 2] = max # Salt
                    else:
                        I_g[i, j, 0] = 0 # Pepper
                        I_g[i, j, 1] = 0 # Pepper
                        I_g[i, j, 2] = 0 # Pepper
    return I_g

def GaussianNoise(I, mean, variance):
    I_g = I

    SD = variance ** 0.5
    # Greyscale
    if I_g.ndim == 2:
        rows, pixs = I_g.shape
        noise = np.random.normal(mean, SD, (rows, pixs))
        noise = noise.reshape(rows, pixs)
        for i in range(I_g.shape[0]):
            for j in range(I_g.shape[1]):
                I_g[i, j] = I_g[i, j] + int(noise[i, j])
    # RGB
    elif I_g.ndim == 3:
        rows, pixs, channels = I_g.shape
        noise = np.random.normal(mean, SD, (rows, pixs, channels))
        noise = noise.reshape(rows, pixs, channels)
        for i in range(I_g.shape[0]):
            for j in range(I_g.shape[1]):
                I_g[i, j, 0] = I_g[i, j, 0] + int(noise[i, j, 0])
                I_g[i, j, 1] = I_g[i, j, 1] + int(noise[i, j, 2])
                I_g[i, j, 2] = I_g[i, j, 1] + int(noise[i, j, 2])
    return I_g

def SpeckleNoise(I):
    I_g = I

    # Greyscale
    if I_g.ndim == 2:
        rows, pixs = I_g.shape
        noise = np.random.randn(rows, pixs)
        noise = noise.reshape(rows, pixs)
        for i in range(I_g.shape[0]):
            for j in range(I_g.shape[1]):
                I_g[i, j] = I_g[i, j] +  int(I_g[i, j] * noise[i, j])
    # RGB
    elif I_g.ndim == 3:
        rows, pixs, channels = I_g.shape
        noise = np.random.randn(rows, pixs, channels)
        noise = noise.reshape(rows, pixs, channels)
        for i in range(I_g.shape[0]):
            for j in range(I_g.shape[1]):
                I_g[i, j, 0] = I_g[i, j, 0] + int(I_g[i, j, 0] * noise[i, j, 0])
                I_g[i, j, 1] = I_g[i, j, 1] + int(I_g[i, j, 1] * noise[i, j, 2])
                I_g[i, j, 2] = I_g[i, j, 1] + int(I_g[i, j, 2] * noise[i, j, 2])
    return I_g

def ImgAverage(Is):
    AvgI = Is[0]
    for i in range(len(Is)):
        if i != 0:
            AvgI += Is[i]
    print('\n\nAVG: ' + str(AvgI) + '\n\n')
    AvgI = AvgI / len(Is)
    return AvgI

    

# Code
# Read and Display Lena Image
imgpath = 'E:/Github Codes and Projects/ImageProcessing_Files/Assignment2/LenaImage.png'
I = cv2.imread(imgpath)
cv2.imshow('Original Image', I)
#cv2.waitKey()

# Convert to GreyScale
I_GreyScale = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
#cv2.imshow('GreyScale Image CV2', I_GreyScale)
# I_GreyScale_Func = rgb2gray(I)
# cv2.imshow('GreyScale', I_GreyScale_Func)

# Add Salt and Pepper Noise
I_SPNoise = SaltPepperNoise(I, 0.1)
#cv2.imshow('Salt and Pepper 0.1', I_SPNoise)

# Add Additive Gaussian Noise
I_GNoise = GaussianNoise(I, 10, 5)
#cv2.imshow('Gaussian Mean 10 Var 5', I_GNoise)

# Add Speckle Noise
I_SNoise = SpeckleNoise(I)
#cv2.imshow('Speckle', I_SNoise)

# Average
Is = [I, I]
I_Avg = ImgAverage(Is)
cv2.imshow('Average', I_Avg)
print(I)
print('\n\n\n')
print(I_Avg)

cv2.waitKey()


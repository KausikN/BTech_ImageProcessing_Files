'''
Take a Lena image and convert it into grayscale. 
Add three different types of noises(salt and pepper, additive Gaussian noise, speckle), 
Take average for each set and display the average images. 
Report the observation made.
'''
import cv2
import numpy as np
import random
import matplotlib.pyplot as plt

def rgb2gray(I):
    I = I.astype(int)
    r, g, b = I[:,:,0], I[:,:,1], I[:,:,2]
    #gray = (0.2989 * r + 0.5870 * g + 0.1140 * b)
    gray = (1 * r + 1 * g + 1 * b) / 3
    return gray.astype(np.uint8)

def SaltPepperNoise(I, prob):
    max = 255
    I_g = I.copy()
    probpercent = int(prob*100)
    # Greyscale
    if I_g.ndim == 2:
        for i in range(I_g.shape[0]):
            for j in range(I_g.shape[1]):
                r = random.randrange(1, 100)
                #r = random.randint(1, 100)
                if r <= probpercent:
                    if r <= int(probpercent / 2):
                        I_g[i, j] = max # Salt
                    else:
                        I_g[i, j] = 0 # Pepper
    # RGB
    elif I_g.ndim == 3:
        for i in range(I_g.shape[0]):
            for j in range(I_g.shape[1]):
                r = random.randrange(1, 100)
                #r = random.randint(1, 100)
                if r <= probpercent:
                    if r <= int(probpercent / 2):
                        I_g[i, j, 0] = max # Salt
                        I_g[i, j, 1] = max # Salt
                        I_g[i, j, 2] = max # Salt
                    else:
                        I_g[i, j, 0] = 0 # Pepper
                        I_g[i, j, 1] = 0 # Pepper
                        I_g[i, j, 2] = 0 # Pepper
    I_g = I_g.astype(np.uint8)
    return I_g

def GaussianNoise(I, mean, variance):
    I_g = I.astype(int).copy()

    SD = variance ** 0.5
    # Greyscale
    if I_g.ndim == 2:
        rows, pixs = I_g.shape
        noise = np.random.normal(mean, SD, (rows, pixs))
        noise = noise.reshape(rows, pixs)
        I_g = np.add(I_g, noise.astype(int))
        # for i in range(I_g.shape[0]):
        #     for j in range(I_g.shape[1]):
        #         I_g[i, j] = I_g[i, j] + int(noise[i, j])
    # RGB
    elif I_g.ndim == 3:
        rows, pixs, channels = I_g.shape
        noise = np.random.normal(mean, SD, (rows, pixs, channels))
        noise = noise.reshape(rows, pixs, channels)
        I_g = np.add(I_g, noise.astype(int))
        # for i in range(I_g.shape[0]):
        #     for j in range(I_g.shape[1]):
        #         I_g[i, j, 0] = I_g[i, j, 0] + int(noise[i, j, 0])
        #         I_g[i, j, 1] = I_g[i, j, 1] + int(noise[i, j, 2])
        #         I_g[i, j, 2] = I_g[i, j, 1] + int(noise[i, j, 2])
    I_g = I_g.astype(np.uint8)
    return I_g

def SpeckleNoise(I):
    I_g = I.astype(int).copy()

    # Greyscale
    if I_g.ndim == 2:
        rows, pixs = I_g.shape
        noise = np.random.randn(rows, pixs)
        noise = noise.reshape(rows, pixs)
        I_g = np.add(I_g, np.multiply(I_g, noise).astype(int))
        # for i in range(I_g.shape[0]):
        #     for j in range(I_g.shape[1]):
        #         I_g[i, j] = I_g[i, j] +  int(I_g[i, j] * noise[i, j])
    # RGB
    elif I_g.ndim == 3:
        rows, pixs, channels = I_g.shape
        noise = np.random.randn(rows, pixs, channels)
        noise = noise.reshape(rows, pixs, channels)
        I_g = np.add(I_g, np.multiply(I_g, noise).astype(int))
        # for i in range(I_g.shape[0]):
        #     for j in range(I_g.shape[1]):
        #         I_g[i, j, 0] = I_g[i, j, 0] + int(I_g[i, j, 0] * noise[i, j, 0])
        #         I_g[i, j, 1] = I_g[i, j, 1] + int(I_g[i, j, 1] * noise[i, j, 2])
        #         I_g[i, j, 2] = I_g[i, j, 1] + int(I_g[i, j, 2] * noise[i, j, 2])
    I_g = I_g.astype(np.uint8)
    return I_g

def ImgAverage(Is):
    AvgI = Is[0].copy().astype(int)
    for imgindex in range(len(Is)):
        if imgindex != 0:
            Is[imgindex] = Is[imgindex].astype(int)
            AvgI = np.add(AvgI, Is[imgindex])
    AvgI = np.divide(AvgI, len(Is)).astype(int)
    AvgI = AvgI.astype(np.uint8)
    return AvgI

    

# Code
# Read and Display Lena Image
plt.figure(figsize=(20,20))

imgpath = 'E:/Github Codes and Projects/ImageProcessing_Files/Assignment2/LenaImage.png'
I = cv2.imread(imgpath)

grayimg = (input("Conv to GreyScale and apply noise? ") in ['y', 'Y', 'YES', 'yes'])
if grayimg:
    I = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)

it = 3
prob = 0.05
mean = 100.0
SD = 50.0
maximginrow = 5

while True:

    choice = input("Enter choice: ")

    if choice in ['g', 'gray', 'grey']:
        # Convert to GreyScale
        I_GreyScale = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
        #cv2.imshow('GreyScale Image CV2', I_GreyScale)
        I_GreyScale_Func = rgb2gray(I)
        #cv2.imshow('GreyScale', I_GreyScale_Func)
        I_app = np.concatenate((I_GreyScale, I_GreyScale_Func), axis=1)
        cv2.imshow("I_APP", I_app)

    elif choice in ['sp', 'saltandpepper', 'SP']:
        it = int(input("Enter no of images to apply noise: "))
        prob = float(input("Enter prob: "))
        nrows = 2 + int(round(it / maximginrow))
        Is = []
        plt.title("Salt and Pepper")
        for i in range(it):
            I_SPNoise = SaltPepperNoise(I, prob)
            plt.subplot(nrows, maximginrow, i+1)
            if not grayimg:
                I_SPNoise = cv2.cvtColor(I_SPNoise, cv2.COLOR_BGR2RGB)
                plt.imshow(I_SPNoise)
            else:
                plt.imshow(I_SPNoise, 'gray')
            Is.append(I_SPNoise)
        I_Avg = ImgAverage(Is)
        ax = plt.subplot(nrows, maximginrow, (nrows-1)*maximginrow + 1)
        ax.title.set_text('Avg')
        if not grayimg:
            plt.imshow(I_Avg)
        else:
            plt.imshow(I_Avg, 'gray')
        plt.show()

    elif choice in ['ga', 'gaussian', 'GA']:
        it = int(input("Enter no of images to apply noise: "))
        mean = float(input("Enter mean: "))
        SD = float(input("Enter SD: "))
        nrows = 1 + int(round(it / maximginrow))
        Is = []
        plt.title("Gaussian")
        for i in range(it):
            I_GNoise = GaussianNoise(I, mean, SD)
            plt.subplot(nrows, maximginrow, i+1)
            if not grayimg:
                I_GNoise = cv2.cvtColor(I_GNoise, cv2.COLOR_BGR2RGB)
                plt.imshow(I_GNoise)
            else:
                plt.imshow(I_GNoise, 'gray')
            Is.append(I_GNoise)
        I_Avg = ImgAverage(Is)
        ax = plt.subplot(nrows, maximginrow, (nrows-1)*maximginrow + 1)
        ax.title.set_text('Avg')
        if not grayimg:
            plt.imshow(I_Avg)
        else:
            plt.imshow(I_Avg, 'gray')
        plt.show()

    elif choice in ['s', 'speckle', 'S']:
        it = int(input("Enter no of images to apply noise: "))
        nrows = 1 + int(round(it / maximginrow))
        Is = []
        plt.title("Speckle")
        for i in range(it):
            I_SNoise = SpeckleNoise(I)
            plt.subplot(nrows, maximginrow, i+1)
            if not grayimg:
                I_SNoise = cv2.cvtColor(I_SNoise, cv2.COLOR_BGR2RGB)
                plt.imshow(I_SNoise)
            else:
                plt.imshow(I_SNoise, 'gray')
            Is.append(I_SNoise)
        I_Avg = ImgAverage(Is)
        ax = plt.subplot(nrows, maximginrow, (nrows-1)*maximginrow + 1)
        ax.title.set_text('Avg')
        if not grayimg:
            plt.imshow(I_Avg)
        else:
            plt.imshow(I_Avg, 'gray')
        plt.show()
    else:
        break
    cv2.waitKey()


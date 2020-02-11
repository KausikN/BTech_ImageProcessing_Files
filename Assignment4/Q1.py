'''
1. Download Lena color image convert it to grayscale image and 
add salt and pepper noise with noise quantity 0.1,0.2 upto 1 and generate 10 noisy images.

    a. Do average filtering ( by correlating with average filter ) of varying sizes for each image. 
    Filter size can be 3*3, 5*5, 7*7. (In 3*3 filter all the values are 1/9, 
    in 5*5 filter all the values are 1/25 and in 7*7 filter all the values are 1/49)

    b. Similarly, repeat the question 1.a by replacing the average filter by median filter.
'''
import cv2
import numpy as np
import random
import matplotlib.pyplot as plt
from tqdm import tqdm

def SaltPepperNoise(I, prob):
    max = 255
    I_g = I.copy()
    probpercent = int(prob*100)
    if I.ndim == 2:
        I_g = np.reshape(I_g, (I_g.shape[0], I_g.shape[1], 1))
    for i in range(I_g.shape[0]):
        for j in range(I_g.shape[1]):
            r = random.randrange(1, 100)
            #r = random.randint(1, 100)
            if r <= probpercent:
                if r <= int(probpercent / 2):
                    I_g[i, j, :] = np.ones(I_g.shape[2]) * max # Salt
                else:
                    I_g[i, j] = np.ones(I_g.shape[2]) * 0 # Pepper
    if I.ndim == 2:
        I_g = np.reshape(I_g, (I_g.shape[0], I_g.shape[1]))
    I_g = I_g.astype(np.uint8)
    return I_g

def Correlation(I, W, stride=(1, 1)):
    I2 = I.copy()
    W = W.copy()
    if I.ndim == 2:
        I2 = np.reshape(I2, (I2.shape[0], I2.shape[1], 1))
        W = np.reshape(W, (W.shape[0], W.shape[1], 1))

    padSize = (I2.shape[0] + 2*(W.shape[0]-1), I2.shape[1] + 2*(W.shape[1]-1), I2.shape[2])
    I_padded = np.zeros(padSize)
    I_padded[W.shape[0]-1:-W.shape[0]+1, W.shape[1]-1:-W.shape[1]+1, :] = I2[:, :, :]

    outSize = (int((I2.shape[0] + W.shape[0])/stride[0]), int((I2.shape[1] + W.shape[1])/stride[1]), I2.shape[2])
    I_g = np.zeros(outSize)

    for i in tqdm(range(0, I_padded.shape[0]-W.shape[0]+1, stride[0])):
        for j in range(0, I_padded.shape[1]-W.shape[1]+1, stride[1]):
            for c in range(I_padded.shape[2]):
                I_g[i, j, c] = np.sum(np.sum(np.multiply(I_padded[i:i+W.shape[0], j:j+W.shape[1], c], W[:, :, c]), axis=1), axis=0)
            
    if I.ndim == 2:
        I_g = np.reshape(I_g, (I_g.shape[0], I_g.shape[1]))
        W = np.reshape(W, (W.shape[0], W.shape[1]))
    I_g = np.round(I_g).astype(np.uint8)
    return I_g

def GenerateEdgeFilter(size=(3, 3)):
    if size[0]%2 == 0 or size[1] == 0:
        return None
    W = np.ones(size) * -1
    W[int(size[0]/2), int(size[1]/2)] = -(size[0]*size[1] - 1)
    return W

def GenerateAverageFilter(size=(3, 3)):
    return np.ones(size).astype(float) / (size[0]*size[1])

def ApplyMedianFilter(I, WSize=(3, 3), stride=(1, 1)):
    I2 = I.copy()
    if I.ndim == 2:
        I2 = np.reshape(I2, (I2.shape[0], I2.shape[1], 1))

    padSize = (I2.shape[0] + 2*(WSize[0]-1), I2.shape[1] + 2*(WSize[1]-1), I2.shape[2])
    I_padded = np.zeros(padSize)
    I_padded[WSize[0]-1:-WSize[0]+1, WSize[1]-1:-WSize[1]+1, :] = I2[:, :, :]

    outSize = (int((I2.shape[0] + WSize[0])/stride[0]), int((I2.shape[1] + WSize[1])/stride[1]), I2.shape[2])
    I_g = np.zeros(outSize)

    for i in tqdm(range(0, I_padded.shape[0]-WSize[0]+1, stride[0])):
        for j in range(0, I_padded.shape[1]-WSize[1]+1, stride[1]):
            for c in range(I_padded.shape[2]):
                I_g[i, j, c] = np.median(I_padded[i:i+WSize[0], j:j+WSize[1], c])
            
    if I.ndim == 2:
        I_g = np.reshape(I_g, (I_g.shape[0], I_g.shape[1]))
    I_g = np.round(I_g).astype(np.uint8)
    return I_g

def ceil(a):
    if (a-float(int(a))) > 0:
        return a + 1
    return a


# Driver Code

# Read Lena Image and convert to Greyscale
I = cv2.imread('Assignment4/LenaImage.png', 0)

# Get Salt and Peppered Images
nCols = 6
nNoisyImgs = 10
nOps = 3
probs = np.arange(1, nNoisyImgs + 1, 1).astype(float) / 10

ax = plt.subplot(1 + int(ceil(nOps*nNoisyImgs/nCols)), nCols, 1)
ax.title.set_text('Original Image')
plt.imshow(I, 'gray')

# A and B

Window = GenerateAverageFilter((3, 3))

NoisyImgs = []
for i in tqdm(range(probs.shape[0])):
    NoisyImgs.append(SaltPepperNoise(I, probs[i]))
    ax = plt.subplot(1 + int(ceil(nOps*nNoisyImgs/nCols)), nCols, 1+nCols + nOps*i)
    ax.title.set_text('Prob ' + str(probs[i]))
    plt.imshow(NoisyImgs[i], 'gray')

    # A
    # Apply Averaging Filter
    I_avg = Correlation(NoisyImgs[i], Window)
    ax = plt.subplot(1 + int(ceil(nOps*nNoisyImgs/nCols)), nCols, 1+nCols + nOps*i+1)
    ax.title.set_text('Averaged ' + str(probs[i]))
    plt.imshow(I_avg, 'gray')

    # B
    # Apply Median Filter
    I_med = ApplyMedianFilter(NoisyImgs[i])
    ax = plt.subplot(1 + int(ceil(nOps*nNoisyImgs/nCols)), nCols, 1+nCols + nOps*i+2)
    ax.title.set_text('Median ' + str(probs[i]))
    plt.imshow(I_med, 'gray')



# ax = plt.subplot(1 + int(ceil(2*nNoisyImgs/nCols)) + 1, nCols, (1 + int(ceil(2*nNoisyImgs/nCols)))*nCols + 1)

plt.show()
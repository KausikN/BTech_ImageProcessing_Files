'''
1. Write a function to implement FFT for 1D signal.
2. Implement DFT function for an image using the FFT for 1D signal using q1.
3. Consider the images of lena and dog images attached. Find phase and magnitude of the dog and lena images using DFT function implemented in q2.
'''
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
import math

def DFT_1D(X, inverse=False):
    invMul = -1
    if inverse:
        invMul *= -1

    N = X.shape[0]
    n = np.arange(N)
    M = np.exp((invMul*2j * np.pi / N) * n)
    return np.dot(M, X)

def IDFT_1D(X):
    return DFT_1D(X, inverse=True) / X.shape[0]

def FFT_1D(X, inverse=False):
    invMul = -1
    if inverse:
        invMul *= -1

    M = X.shape[0]
    #print(M)

    if M == 1:
        return DFT_1D(X, inverse)
    elif M % 2 > 0:
        print("size of X must be a power of 2")
        return None
    else:
        X_even = FFT_1D(X[::2], inverse) # Even
        X_odd = FFT_1D(X[1::2], inverse) # Odd
        coeffs = np.exp((invMul*2j * np.pi / M) * np.arange(M))
        #print("DEBUG:", coeffs[ : math.ceil(M/2)].shape, coeffs[math.ceil(M/2) : ].shape, X_odd.shape, X_even.shape)
        return np.concatenate([X_even + (coeffs[ : math.ceil(M/2)] * X_odd),
                               X_even + (coeffs[math.ceil(M/2) : ] * X_odd)])

def IFFT_1D(X):
    return FFT_1D(X, inverse=True) / X.shape[0]

def FFT_2D(X):
    X_FFT = X.copy().astype(np.complex)
    # Apply FFT along rows
    for i in tqdm(range(X_FFT.shape[0])):
        X_FFT[i, :] = FFT_1D(X_FFT[i, :])
    # Apply FFT along columns
    for j in tqdm(range(X_FFT.shape[1])):
        X_FFT[:, j] = FFT_1D(X_FFT[:, j])
    return X_FFT

def IFFT_2D(X):
    I_IFFT = X.copy().astype(np.complex)
    # Apply IFFT along columns
    for j in tqdm(range(I_IFFT.shape[1])):
        I_IFFT[:, j] = IFFT_1D(I_IFFT[:, j])
    # Apply IFFT along rows
    for i in tqdm(range(I_IFFT.shape[0])):
        I_IFFT[i, :] = IFFT_1D(I_IFFT[i, :])
    return I_IFFT.astype(np.uint8)

def Conv2Polar(I_FFT):
    I_Mag = I_FFT.copy().astype(float)
    I_Phase = I_FFT.copy().astype(float)
    for i in tqdm(range(I_FFT.shape[0])):
        I_Mag[i, :] = GetMagnitude(I_FFT[i, :])
        I_Phase[i, :] = GetPhase(I_FFT[i, :])
    return I_Mag, I_Phase

def Conv2Cartesian(I_Mag, I_Phase):
    I_FFT = I_Mag.copy().astype(np.complex)
    for i in tqdm(range(I_FFT.shape[0])):
        for j in range(I_FFT.shape[1]):
            I_FFT[i, j] = I_Mag[i, j] * np.exp(1j * I_Phase[i, j])
    return I_FFT

def GetMagnitude(compVal):
    return np.abs(compVal)

def GetPhase(compVal):
    return np.angle(compVal)

# Driver Code
# 1
# Implemented FFT

# 2
# Read Lena Image
imgPath = 'Assignment5/Lena.png'
I = cv2.imread(imgPath, 0)
#I = np.array([[1, 2, 3, 4]])
print(I.shape)
I_FFT = FFT_2D(I)

print("FFT", I_FFT)

# 3
# Get Magnitude and Phase
print("Converting to Polar Coor...")
I_Mag, I_Phase = Conv2Polar(I_FFT)
I_FFT_Cart = Conv2Cartesian(I_Mag, I_Phase)

print("Mag:", I_Mag)
print("Phase:", I_Phase)

I_FFT = IFFT_2D(I_FFT)

plt.imshow(I_FFT.astype(np.uint8), 'gray')
plt.show()

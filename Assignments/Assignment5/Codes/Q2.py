'''
4. Swap phase of the dog image and magnitude of the lena image and display the output.
5. Swap phase of the lena image and magnitude of the dog image ad display the output.
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
    X_IFFT = X.copy().astype(np.complex)
    # Apply IFFT along columns
    for j in tqdm(range(X_IFFT.shape[1])):
        X_IFFT[:, j] = IFFT_1D(X_IFFT[:, j])
    # Apply IFFT along rows
    for i in tqdm(range(X_IFFT.shape[0])):
        X_IFFT[i, :] = IFFT_1D(X_IFFT[i, :])
    return X_IFFT.astype(np.uint8)

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
# 4 and 5
dogPath = 'Assignment5/dog2.png'
lenaPath = 'Assignment5/Lena.png'
dogI = cv2.imread(dogPath, 0)
LenaI = cv2.imread(lenaPath, 0)

print("Dog:", dogI.shape)
print("Lena:", LenaI.shape)

# Get Mags and Phases
print("Dog FFT...")
dogI_FFT = FFT_2D(dogI)
print("Dog Conv 2 Polar...")
dogI_Mag, dogI_Phase = Conv2Polar(dogI_FFT)

print("Lena FFT...")
LenaI_FFT = FFT_2D(LenaI)
print("Lena Conv 2 Polar...")
LenaI_Mag, LenaI_Phase = Conv2Polar(LenaI_FFT)

print("Cartesian Switching...")
dogMagLenaPhase = Conv2Cartesian(dogI_Mag, LenaI_Phase)
LenaMagdogPhase = Conv2Cartesian(LenaI_Mag, dogI_Phase)

print("IFFT...")
dogMagLenaPhase_I = IFFT_2D(dogMagLenaPhase)
LenaMagdogPhase_I = IFFT_2D(LenaMagdogPhase)

plt.subplot(2, 2, 1)
plt.imshow(dogI.astype(np.uint8), 'gray')
plt.subplot(2, 2, 2)
plt.imshow(LenaI.astype(np.uint8), 'gray')
plt.subplot(2, 2, 3)
plt.imshow(dogMagLenaPhase_I.astype(np.uint8), 'gray')
plt.subplot(2, 2, 4)
plt.imshow(LenaMagdogPhase_I.astype(np.uint8), 'gray')
plt.show()
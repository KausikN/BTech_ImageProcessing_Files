'''
3. Do histogram equalization on pout-dark and display the same
4. Do histogram matching(specification) on the pout-dark image, keeping pout-bright as a reference image.
(Please find the attached pout-dark,pout-bright images)
'''
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
from tqdm import tqdm


# 3 and 4
def HistogramMatching(I_input, I_ref, pixelRange=(0, 255)):
    HistVals_input, HistProbDist_input, Hist2PixVals_input = HistogramEqualisation(I_input)
    HistVals_ref, HistProbDist_ref, Hist2PixVals_ref = HistogramEqualisation(I_ref)

    ProcessedDist = []
    noMapCheck = True
    for hv in tqdm(HistVals_input):
        if len(Hist2PixVals_ref[hv - pixelRange[0]]) > 0:
            randindex = random.randint(0, len(Hist2PixVals_ref[hv - pixelRange[0]])-1)
            ProcessedDist.append(Hist2PixVals_ref[hv - pixelRange[0]][randindex])
            noMapCheck = False
        else:
            ProcessedDist.append(-1)
            # if len(ProcessedDist) > 0:
            #     ProcessedDist.append(ProcessedDist[-1])
            # else:
            #     ProcessedDist.append(ProcessedDist[-1])
    
    if not noMapCheck:
        while (-1 in ProcessedDist):
            if ProcessedDist[0] == -1:
                ProcessedDist[0] = ProcessedDist[1]
            if ProcessedDist[-1] == -1:
                ProcessedDist[-1] = ProcessedDist[-2]
            
            for pdi in range(1, len(ProcessedDist)-1):
                if ProcessedDist[pdi] == -1:
                    if ProcessedDist[pdi-1] != -1 and ProcessedDist[pdi+1] != -1:
                        ProcessedDist[pdi] = int(round((ProcessedDist[pdi-1] + ProcessedDist[pdi+1]) / 2))
                    elif ProcessedDist[pdi-1] == -1 and ProcessedDist[pdi+1] != -1:
                        ProcessedDist[pdi] = ProcessedDist[pdi+1]
                    elif ProcessedDist[pdi-1] != -1 and ProcessedDist[pdi+1] == -1:
                        ProcessedDist[pdi] = ProcessedDist[pdi-1]

        ProbDist_input = np.array(GetFreqDist(I_input, pixelRange)) / (I_input.shape[0]*I_input.shape[1])
        ProbDist_processed = []
        for i in range(len(ProcessedDist)):
            ProbDist_processed.append(0.0)
        for pi in range(ProbDist_input.shape[0]):
            ProbDist_processed[ProcessedDist[pi] - pixelRange[0]] += ProbDist_input
        ProbDist_processed = np.array(ProbDist_processed)

        return ProcessedDist, ProbDist_processed
        
    print("No Mapping Exists for Reference to Input Image")
    return None

def HistogramEqualisation(Image, pixelRange=(0, 255)):
    FreqDist = np.array(GetFreqDist(Image, pixelRange))
    TotPixels = Image.shape[0]*Image.shape[1]

    ProbDist = (FreqDist / TotPixels)

    CumulativeProbDist = GetCumulativeDist(ProbDist)

    HistVals = np.round(CumulativeProbDist * (pixelRange[1] - pixelRange[0])).astype(int)

    HistProbDist = []
    Hist2PixVals = []
    for i in range(pixelRange[0], pixelRange[1]+1):
        HistProbDist.append(0.0)
        Hist2PixVals.append([])
    for hvi in range(HistVals.shape[0]):
        HistProbDist[HistVals[hvi] - pixelRange[0]] += ProbDist[hvi]
        Hist2PixVals[HistVals[hvi] - pixelRange[0]].append(hvi + pixelRange[0])
    HistProbDist = np.array(HistProbDist)

    return HistVals, HistProbDist, Hist2PixVals

def GetFreqDist(Image, pixelRange=(0, 255)):
    Freq = []
    for i in range(pixelRange[0], pixelRange[1]+1):
        Freq.append(0)
    for row in tqdm(Image):
        for pixel in row:
            Freq[pixel - pixelRange[0]] += 1
    return Freq

def GetCumulativeDist(Dist):
    CumulativeDist = []
    cumulativeVal = 0.0
    for d in Dist:
        cumulativeVal += d
        CumulativeDist.append(cumulativeVal)
    return np.array(CumulativeDist)

def HistPlot(Data, nbins=25):
    X = np.arange(len(Data))
    n, bins, patches = plt.hist(Data, nbins, facecolor='blue', alpha=0.5)
    #plt.show()

def ImagePixelReplace(Image, pixelReplaceVals):
    I = np.zeros(Image.shape)
    for i in tqdm(range(Image.shape[0])):
        for j in range(Image.shape[1]):
            I[i, j] = pixelReplaceVals[Image[i, j]]
    return I
'''
'''
# Driver Code - 3
imgPath = 'Assignment3/pout-dark.jpg'
# imgPath = 'Assignment3/Curtain_Dark.jpg'
I = cv2.imread(imgPath, 0)
HistVals, HistProbDist, Hist2PixVals = HistogramEqualisation(I)

plt.subplot(2, 2, 1)
plt.imshow(I, 'gray')
plt.subplot(2, 2, 2)
plt.imshow(ImagePixelReplace(I, HistVals), 'gray')
# plt.subplot(2, 2, 3)
# HistPlot(HistVals)
# plt.subplot(2, 2, 4)
# HistPlot(HistProbDist)
plt.show()

print("HistVals:", HistVals)
print("HistProbDist:", HistProbDist)
'''
'''
# Driver Code - 4
imgPath_input = 'Assignment3/pout-dark.jpg'
imgPath_ref = 'Assignment3/pout-bright.jpg'
# imgPath_input = 'Assignment3/Curtain_Dark.jpg'
# imgPath_ref = 'Assignment3/pout-bright.jpg'
I_input = cv2.imread(imgPath_input, 0)
I_ref = cv2.imread(imgPath_ref, 0)

ProcessedDist, ProbDist_processed = HistogramMatching(I_input, I_ref)
I_processed = ImagePixelReplace(I_input, ProcessedDist)

ax = plt.subplot(2, 3, 1)
plt.imshow(I_ref, 'gray')
ax.title.set_text('Reference')
ax = plt.subplot(2, 3, 2)
plt.imshow(I_input, 'gray')
ax.title.set_text('Input')
ax = plt.subplot(2, 3, 3)
plt.imshow(I_processed, 'gray')
ax.title.set_text('HistMatched')
# ax = plt.subplot(2, 3, 4)
# HistPlot(ProcessedDist)
# ax.title.set_text('HistMatchMap')
# ax = plt.subplot(2, 3, 5)
# HistPlot(ProbDist_processed)
# ax.title.set_text('HistMatchProb')

# print("ProcessedDist:", ProcessedDist)
# print("ProbDist_processed:", ProbDist_processed)
plt.show()
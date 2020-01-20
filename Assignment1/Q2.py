'''
2. Read the matrix size through keyboard and create a random matrix of integers ranging from  0 to 10 and compute all the above functions listed in question 1
'''

import random

# Creates a matrix with user input
def CreateMatrix_RandomInput(matrixSize, ValRange):
    matrix = []

    for i in range(matrixSize[0]):
        row = []
        for j in range(matrixSize[1]):
            row.append(random.randint(ValRange[0], ValRange[1]))
        matrix.append(row)
    
    return matrix

def MatSum(matrix):
    sum = 0

    for row in matrix:
        for val in row:
            sum += val
    
    return sum

def MatMax(matrix):
    max = 0

    for row in matrix:
        for val in row:
            if max < val:
                max = val
    
    return max

def MatFreqDist(matrix):
    freq = {}

    for row in matrix:
        for val in row:
            freq[val] = 0

    for row in matrix:
        for val in row:
            freq[val] += 1

    return freq

def MatMean(matrix):
    return MatSum(matrix) / (len(matrix) * len(matrix[0]))

def MatMedian(matrix):
    arr = []
    for row in matrix:
        for val in row:
            arr.append(val)
    BubbleSort(arr)

    if len(arr) % 2 == 1:
        return arr[int((len(arr) - 1)/2)]
    else:
        return (arr[int(len(arr)/2)] + arr[int(len(arr)/2 - 1)]) / 2

def MatMode(matrix):
    modeVal = -1
    modeVal_freq = -1

    freq = MatFreqDist(matrix)

    for key in freq.keys():
        if freq[key] > modeVal_freq:
            modeVal = key
            modeVal_freq = freq[key]

    return modeVal

def MatStandardDeviation(matrix):
    SD = 0.0

    mean = MatMean(matrix)
    sumsqaurediff = 0.0
    for row in matrix:
        for val in row:
            sumsqaurediff += (float(val) - mean) ** 2
    SD = (sumsqaurediff / (len(matrix) * len(matrix[0]))) ** (0.5)

    return SD


def BubbleSort(arr):
    n = len(arr)
 
    # Traverse through all array elements
    for i in range(n):
 
        # Last i elements are already in place
        for j in range(0, n-i-1):
 
            # traverse the array from 0 to n-i-1
            # Swap if the element found is greater
            # than the next element
            if arr[j] > arr[j+1] :
                arr[j], arr[j+1] = arr[j+1], arr[j]


# Driver Code
matrixRows = int(input("Enter no of Rows: "))
matrixColumns = int(input("Enter no of Columns: "))
matrixSize = (matrixRows, matrixColumns)
ValRange = (1, 10)
matrix = CreateMatrix_RandomInput(matrixSize, ValRange)
print("Input Matrix: ")
for row in matrix:
    print(row)
print("Sum:", MatSum(matrix))
print("Max:", MatMax(matrix))
print("Mean:", MatMean(matrix))
print("Median:", MatMedian(matrix))
print("Mode:", MatMode(matrix))
print("Standard Deviation:", MatStandardDeviation(matrix))

freqdist = MatFreqDist(matrix)
print("Frequency Distribution:")
for key in freqdist.keys():
    print(key, ": ", freqdist[key])

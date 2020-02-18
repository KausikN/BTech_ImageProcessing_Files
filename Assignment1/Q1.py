'''
1. Read a matrix of size 5*5 and find the following by using a user-defined function.
    i.   sum
    ii.  maximum
    iii. mean  
    iv.  median   
    v.   mode    
    vi.  standard deviation       
    v.   frequency distribution
'''

def CreateMatrix_UserInput(matrixSize):
    ''' Creates a matrix with user input '''
    matrix = []

    for i in range(matrixSize[0]):
        row = []
        for j in range(matrixSize[1]):
            row.append(int(input("Enter value at (" + str(i) + ", " + str(j) + "): ")))
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
    sum = 0
    for row in matrix:
        for val in row:
            sum += val
    return sum / (len(matrix) * len(matrix[0]))

def MatMedian(matrix):
    arr = []
    for row in matrix:
        for val in row:
            arr.append(val)
    # Bubble Sort Code
    n = len(arr)
    for i in range(n):
        for j in range(0, n-i-1):
            if arr[j] > arr[j+1] :
                arr[j], arr[j+1] = arr[j+1], arr[j]

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
            sumsqaurediff += (val - mean) ** 2
    SD = (sumsqaurediff / (len(matrix) * len(matrix[0]))) ** (1/2)

    return SD





# Driver Code
matrixSize = (5, 5)
matrix = CreateMatrix_UserInput(matrixSize)
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

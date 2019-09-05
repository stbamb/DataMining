import math

_attr_keyword = '@attribute'
_data_folder = 'KEEL_data'

def main():
    #askForInput()
    k = 15
    fileName = "titanic-10-fold/titanic-10"
    trainingDataSet = readData(fileName, True)
    testDataSet = readData(fileName, False)
    print("testDataSet:", testDataSet)
    print("trainingDataSet:", trainingDataSet)
    testDataSetResults = cosDistance1(trainingDataSet, testDataSet)
    trainingDataSetResults = cosDistance2(trainingDataSet)
    print("testDataSetResults:", testDataSetResults)
    print("trainingDataSetResults:", trainingDataSetResults)
    rightPredictions = nearestNeighbor(trainingDataSetResults, testDataSetResults, k)
    print("rightPredictions:", rightPredictions, "out of", len(testDataSet))
    accuracy = (rightPredictions / len(testDataSet)) * 100
    print("accuracy:", accuracy)


def askForInput():
    fileName = input("Please enter the name of the dataset: ")
    k = input("Please enter the value for k: ")


def readData(fileName, isTrainingData):
    dataSet = []

    if isTrainingData:
        fileName += "-1tra.dat"
    else:
        fileName += "-1tst.dat"

    with open(fileName) as f:
        datafile = f.readlines()
        for line in datafile:
            if line[0] != '@':
                data = [float(x) for x in line.split(",")]
                dataSet.append(data)

    return dataSet


def cosDistance1(trainingDataSet, testDataSet):
    cosSet = []

    for i in range(len(testDataSet)):
        A = testDataSet[i]
        for j in range(len(trainingDataSet)):
            B = trainingDataSet[j]
            cos = 0
            numerator = 0
            denominator1 = 0
            denominator2 = 0

            for k in range(len(A) - 1):
                numerator += (A[k] * B[k])
                denominator1 += pow(A[k], 2)
                denominator2 += pow(B[k], 2)

            cos += numerator / (math.sqrt(denominator1) * math.sqrt(denominator2))
        cosSet.append((cos/len(trainingDataSet), testDataSet[i][-1]))

    return cosSet


def cosDistance2(dataSet):
    cosSet = []

    for i in range(len(dataSet)):
        A = dataSet[i]
        for j in range(len(dataSet)):
            if i == j:
                continue
            else:
                B = dataSet[j]
                cos = 0
                numerator = 0
                denominator1 = 0
                denominator2 = 0

                for k in range(len(A) - 1):
                    numerator += (A[k] * B[k])
                    denominator1 += pow(A[k], 2)
                    denominator2 += pow(B[k], 2)

                cos += numerator / (math.sqrt(denominator1) * math.sqrt(denominator2))
        cosSet.append((cos/len(dataSet), dataSet[i][-1]))

    return cosSet


def nearestNeighbor(trainingDataSetResults, testDataSetResults, k):
    distances = calculateDistances(trainingDataSetResults, testDataSetResults, k)
    rightPredictions = 0
    print("distances:", distances)
    print(len(distances))

    for i in range(len(distances)):
        instance = distances[i]
        expected = instance[0][1]
        count1 = 0
        count2 = 0
        for j in range(len(instance)):
            if instance[j][2] == expected:
                count1 += 1
            else:
                count2 += 1
        if count1 > count2:
            rightPredictions += 1

    return rightPredictions


def calculateDistances(trainingDataSetResults, testDataSetResults, k):
    distances = []
    for i in range(len(testDataSetResults)):
        kDistances = []
        for j in range(len(trainingDataSetResults)):
            dist = abs(testDataSetResults[i][0] - trainingDataSetResults[j][0])
            kDistances.append((dist, testDataSetResults[i][1], trainingDataSetResults[j][1]))
        sortedDistances = mergesort(kDistances)
        distances.append(sortedDistances[0:k])
    return distances


def mergesort(arr):
    n = len(arr)
    if n == 1:
        return arr
    arr1 = arr[0:n // 2]
    arr2 = arr[n // 2:]
    arr1 = mergesort(arr1)
    arr2 = mergesort(arr2)
    return mergesort_aux(arr1, arr2)


def mergesort_aux(arr1, arr2):
    result = []
    n1 = len(arr1)
    n2 = len(arr2)
    while n1 > 0 and n2 > 0:
        if arr1[0] > arr2[0]:
            result.append(arr2[0])
            del arr2[0]
            n2 -= 1
        else:
            result.append(arr1[0])
            del arr1[0]
            n1 -= 1
    while n1 > 0:
        result.append(arr1[0])
        del arr1[0]
        n1 -= 1
    while n2 > 0:
        result.append(arr2[0])
        del arr2[0]
        n2 -= 1
    return result


main()

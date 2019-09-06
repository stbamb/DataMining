import math

_data_folder = 'KEEL_data'


def main():
    # askForInput()
    k = 3
    fileName = "pima-10-fold/pima-10"
    trainingDataSet = readData(fileName, True)
    testDataSet = readData(fileName, False)
    print("testDataSet:", testDataSet)
    print("trainingDataSet:", trainingDataSet)
    # distances = cosDistance(trainingDataSet, testDataSet)
    # print(len(distances))
    # print(len(distances[0]))
    # print("distances:", distances)

    # rightPredictions = nearestNeighbor(distances, k)
    # print("rightPredictions:", rightPredictions, "out of", len(testDataSet))
    # accuracy = (rightPredictions / len(testDataSet)) * 100
    # print("accuracy:", accuracy)


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
                try:
                    data = [float(x) for x in line.split(",")]
                except:
                    data = line.split(",")
                    for elem in data[:8]:
                        #if not elem == data[-1]:
                            data = float(elem)
                dataSet.append(data)

    return dataSet


def cosDistance(trainingDataSet, testDataSet):
    cosDistances = []
    for i in range(len(testDataSet)):
        distances = []
        A = testDataSet[i]
        for j in range(len(trainingDataSet)):
            B = trainingDataSet[j]
            numerator = 0
            denominator1 = 0
            denominator2 = 0

            for k in range(len(A) - 1):
                numerator += (A[k] * B[k])
                denominator1 += pow(A[k], 2)
                denominator2 += pow(B[k], 2)

            cosDist = numerator / (math.sqrt(denominator1) * math.sqrt(denominator2))
            distances.append((cosDist, testDataSet[i][-1], trainingDataSet[j][-1]))
        cosDistances.append(distances)

    return cosDistances


def nearestNeighbor(distances, k):
    rightPredictions = 0

    for i in range(len(distances)):
        distances[i] = mergesort(distances[i])
        distances[i] = distances[i][0:k]
    print("distances:", distances)
    omd = []
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
        if count1 >= count2:
            rightPredictions += 1

            omd.append(i)

    print("corrected predicted", omd)
    return rightPredictions


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

import math

_k = 0
_file_name = ''
_data_folder = 'KEEL_data/'
_invalid_option_msg = "That is not a valid option. Try again.\n"


def main():
    option = menu()
    #fileName = _data_folder + "pima-10-fold/pima-10"   # used only for debug purposes

    if option == 1:
        askForInput()

    print(_file_name)
    trainingDataSet = readData(_file_name, True)
    testDataSet = readData(_file_name, False)

    if trainingDataSet == [] or testDataSet == []:
        print("One or more files could not be found.\n")
        return

    print("testDataSet:", testDataSet)
    print("trainingDataSet:", trainingDataSet)
    distances = cosDistance(trainingDataSet, testDataSet)
    print(len(distances))
    print(len(distances[0]))

    rightPredictions = nearestNeighbor(distances, _k)
    print("rightPredictions:", rightPredictions, "out of", len(testDataSet))
    accuracy = (rightPredictions / len(testDataSet)) * 100
    print("accuracy:", accuracy)


def askForInput():
    validOption = False
    while not validOption:
        try:
            global _file_name
            global _k
            _file_name = input("Enter the name of the dataset you want to use: ")
            _k = int(input("Enter the value of k: "))
            validOption = not validOption
        except:
            print(_invalid_option_msg)


def readData(fileName, isTrainingData):
    dataSet = []
    if isTrainingData:
        fileName += "-1tra.dat"
    else:
        fileName += "-1tst.dat"
    try:
        with open(fileName) as f:
            datafile = f.readlines()
            for line in datafile:
                if line[0] != '@':
                    try:
                        data = [float(x) for x in line.split(",")]
                    except ValueError:
                        data = line.split(",")
                        for i in range(len(data) - 1):
                            data[i] = float(data[i])
                    dataSet.append(data)
    except:
        return []
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
    print("sortedDistances:", distances)
    correctlyPredicted = []
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

            correctlyPredicted.append(i)

    print("correctly predicted", correctlyPredicted)
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


def menu():
    option = 0
    validOption = False
    while not validOption:
        try:
            print("Experimenter. Please select an option:")
            print("1. Run knn on a specific problem set")
            print("2. Explore k values")
            print("3. Run on all datasets")
            option = int(input())
            if 0 < option < 4:
                validOption = not validOption
            else:
                print(_invalid_option_msg)
        except ValueError:
            print(_invalid_option_msg)
    return option


main()

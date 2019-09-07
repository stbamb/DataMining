import math

_k = 0
_file_name = ""
_data_folder = "KEEL_data/"
_invalid_option_msg = "That is not a valid option. Try again.\n"
_max_files_per_fold = 10
_default_distance_algorithm = "manhattanDistance"
_data_folder_names = ["appendicitis-10-fold", "banana-10-fold", "magic-10-fold", "phoneme-10-fold",
                      "pima-10-fold", "ring-10-fold", "sonar-10-fold", "spambase-10-fold",
                      "titanic-10-fold", "twonorm-10-fold", "wdbc-10-fold"]


def main():
    option = menu()
    askForInput(option)
    if option == 1:
        runForSpecificK()
    elif option == 2:
        runForAllK()
    elif option == 3:
        runForAll()


def runForSpecificK():
    overallAccuracy = 0
    for i in range(1, _max_files_per_fold + 1):
        trainingDataSet = readData(_file_name, True, i)
        testDataSet = readData(_file_name, False, i)

        if trainingDataSet == [] or testDataSet == []:
            return "One or more files could not be found.\n"

        distances = calculateDistance(trainingDataSet, testDataSet)
        #print("testDataSet:", testDataSet)
        #print("trainingDataSet:", trainingDataSet)
        #print(distances)
        #print(len(distances))
        #print(len(distances[0]))

        rightPredictions = nearestNeighbor(distances, _k)
        print("rightPredictions:", rightPredictions, "out of", len(testDataSet), "with k =", str(_k))
        accuracy = (rightPredictions / len(testDataSet)) * 100
        print("accuracy:", accuracy, "with k =", str(_k))
        print()
        overallAccuracy += accuracy

    overallAccuracy /= _max_files_per_fold
    print("overallAccuracy:", overallAccuracy)


def runForAllK():
    for k in range(1, 11):
        overallAccuracy = 0
        for i in range(1, _max_files_per_fold + 1):

            print("Testing for file", i, "of", _file_name)

            trainingDataSet = readData(_file_name, True, i)
            testDataSet = readData(_file_name, False, i)

            if trainingDataSet == [] or testDataSet == []:
                return "One or more files could not be found.\n"

            distances = calculateDistance(trainingDataSet, testDataSet)
            # print("testDataSet:", testDataSet)
            # print("trainingDataSet:", trainingDataSet)
            # print(distances)
            # print(len(distances))
            # print(len(distances[0]))

            rightPredictions = nearestNeighbor(distances, k)
            print("rightPredictions:", rightPredictions, "out of", len(testDataSet), "with k =", str(k))
            accuracy = (rightPredictions / len(testDataSet)) * 100
            print("accuracy:", accuracy, "with k =", str(k))
            print()
            overallAccuracy += accuracy

        overallAccuracy /= _max_files_per_fold
        print("overallAccuracy: " + str(overallAccuracy))


def runForAll():
    global _file_name
    for i in range(len(_data_folder_names)):
        _file_name = _data_folder + "/" + _data_folder_names[i] + "/" + getFileName(_data_folder_names[i])
        print(_file_name)
        overallAccuracy = 0
        for j in range(1, _max_files_per_fold + 1):
            trainingDataSet = readData(_file_name, True, j)
            testDataSet = readData(_file_name, False, j)

            if trainingDataSet == [] or testDataSet == []:
                return "One or more files could not be found.\n"

            distances = calculateDistance(trainingDataSet, testDataSet)
            # print("testDataSet:", testDataSet)
            # print("trainingDataSet:", trainingDataSet)
            # print(distances)
            # print(len(distances))
            # print(len(distances[0]))

            rightPredictions = nearestNeighbor(distances, _k)
            print("rightPredictions:", rightPredictions, "out of", len(testDataSet), "with k =", str(_k))
            accuracy = (rightPredictions / len(testDataSet)) * 100
            print("accuracy:", accuracy, "with k =", str(_k))
            print()
            overallAccuracy += accuracy

        overallAccuracy /= _max_files_per_fold
        print("overallAccuracy:", overallAccuracy)


def askForInput(option):
    validOption = False
    global _file_name, _data_folder, _k
    if option == 1:
        while not validOption:
            try:
                _file_name = input("Enter the name of the dataset you want to use: ")
                _k = int(input("Enter the value of k: "))
                validOption = not validOption
            except:
                print(_invalid_option_msg)
    elif option == 2:
        while not validOption:
            try:
                _file_name = input("Enter the name of the dataset you want to use: ")
                validOption = not validOption
            except:
                print(_invalid_option_msg)
    elif option == 3:
        while not validOption:
            try:
                _data_folder = input("Enter the name of the folder containing the subfolders for the folds: ")
                _k = int(input("Enter the value of k: "))
                validOption = not validOption
            except:
                print(_invalid_option_msg)


def readData(fileName, isTrainingData, i):
    dataSet = []
    if isTrainingData:
        fileName += "-" + str(i) + "tra.dat"
    else:
        fileName += "-" + str(i) + "tst.dat"
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


def calculateDistance(trainingDataSet, testDataSet):
    return manhattanDistance(trainingDataSet, testDataSet)


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


def manhattanDistance(trainingDataSet, testDataSet):
    manhattanDistances = []

    for i in range(len(testDataSet)):
        A = testDataSet[i]
        distances = []
        for j in range(len(trainingDataSet)):
            B = trainingDataSet[j]
            dist = 0
            for k in range(len(A) - 1):
                dist += abs(A[k] - B[k])
            distances.append((dist, testDataSet[i][-1], trainingDataSet[j][-1]))
        manhattanDistances.append(distances)

    return manhattanDistances


def nearestNeighbor(distances, k):
    rightPredictions = 0

    for i in range(len(distances)):
        distances[i] = mergesort(distances[i])
        distances[i] = distances[i][0:k]
    #print("sortedDistances:", distances)
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

    #print("correctly predicted", correctlyPredicted)
    return rightPredictions


def getFileName(folderName):
    names = folderName.split("-")
    return names[0] + "-" + names[1]


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

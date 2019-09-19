# author        : Esteban
# course        : CS-691 Data Mining
# name          : inhouse_knn.py
# date          : 20190920
# python_version: 3.7
# notes         : Assignment2
# description   : Needed for esteban_murillo_assignment2.py to run
# ==============================================================================

import math

import global_variables as global_vars


def calculateDistance(default_distance_algorithm, *training_data_set, **test_data_set):
    return default_distance_algorithm(*training_data_set, **test_data_set)


def cosDistance(training_data_set, test_data_set):
    cos_distances = []

    for i in range(len(test_data_set)):
        distances = []
        A = test_data_set[i][0]
        for j in range(len(training_data_set)):
            B = training_data_set[j][0]
            numerator = 0
            denominator1 = 0
            denominator2 = 0

            for k in range(len(A) - 1):
                numerator += (A[k] * B[k])
                denominator1 += pow(A[k], 2)
                denominator2 += pow(B[k], 2)

            cos_dist = numerator / (math.sqrt(denominator1) * math.sqrt(denominator2))
            distances.append((cos_dist, test_data_set[i][-1], training_data_set[j][-1]))
        cos_distances.append(distances)

    return cos_distances


def manhattanDistance(training_data_set, test_data_set):
    manhattan_distances = []

    for i in range(len(test_data_set)):
        A = test_data_set[i][0]
        distances = []
        for j in range(len(training_data_set)):
            B = training_data_set[j][0]
            dist = 0
            for k in range(len(A) - 1):
                dist += abs(A[k] - B[k])
            distances.append((dist, test_data_set[i][-1], training_data_set[j][-1]))
        manhattan_distances.append(distances)

    return manhattan_distances


def kNN(distances, k):
    right_predictions = 0

    for i in range(len(distances)):
        distances[i] = mergesort(distances[i])
        distances[i] = distances[i][0:k]
    correctly_predicted = []
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
            right_predictions += 1

            correctly_predicted.append(i)

    if global_vars.debug:
        print("correctly predicted", correctly_predicted)
    return right_predictions


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

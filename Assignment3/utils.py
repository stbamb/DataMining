# author        : Esteban
# course        : CS-691 Data Mining
# name          : utils.py
# date          : 20191005
# python_version: 3.7
# notes         : Assignment3
# description   : Needed for esteban_murillo_assignment3.py to run
# ==============================================================================
import os
import random
from itertools import combinations

import global_variables


def getAllReviewsText(reviews_files_names):
    documents_text = []
    for fileName in reviews_files_names:
        text_array = []
        try:
            with open(fileName) as f:
                text = f.read().replace('\n', '')
                f.close()
                text_array.append(text)
                documents_text.append(text_array)
        except FileNotFoundError:
            print("Error reading", fileName)
        except UnicodeDecodeError:
            print("Invalid character in", fileName)
    return documents_text


def getReviewFilesNames(folder1, folder2):
    pos_reviews_files_names = []
    neg_reviews_files_names = []
    directory = os.fsencode(folder1)

    try:
        for file in os.listdir(directory):
            file_name = os.fsdecode(file)
            if file_name.endswith(global_variables.reviews_file_extension):
                pos_reviews_files_names.append(folder1 + file_name)

        directory = os.fsencode(folder2)

        for file in os.listdir(directory):
            file_name = os.fsdecode(file)
            if file_name.endswith(global_variables.reviews_file_extension):
                neg_reviews_files_names.append(folder2 + file_name)

    except FileNotFoundError:
        print("Directory not found")
        exit(0)

    return pos_reviews_files_names, neg_reviews_files_names


def partitionDataSet(images_names):
    size_per_fold = len(images_names) / global_variables.x_fold
    partitioned_list = []
    sub_fold = []

    for i in range(len(images_names)):
        sub_fold.append(images_names[i])
        if i % size_per_fold == size_per_fold - 1:
            partitioned_list.append(sub_fold)
            sub_fold = []

    return partitioned_list


def addLabels(arr, is_pos):
    ultimate_arr = []
    for element in arr:
        if is_pos:
            ultimate_arr.append((element, 1))
        else:
            ultimate_arr.append((element, -1))
    return ultimate_arr


def getFoldsCombinations(partitioned_data):
    complete_set = []
    broken_down_set = []
    comb = combinations(partitioned_data, global_variables.x_fold - 1)
    for i in list(comb):
        complete_set.append(i)

    for i in range(len(partitioned_data)):
        complete_set.append(partitioned_data[i])

    #print(complete_set)
    for i in range(len(complete_set)):
        for j in range(len(complete_set[i])):
            broken_down_set.append(complete_set[i][j])

    #print(broken_down_set)
    return complete_set


def joinTrainSet(train_set):
    joined_train_set = []
    for i in range(len(train_set)):
        for j in range(len(train_set[i])):
            joined_train_set.append(train_set[i][j])
    return joined_train_set


def shuffle(arr):
    for i in range(len(arr) - 1, -1, -1):
        random_index = random.randint(0, len(arr) - 1)
        temp = arr[random_index]
        arr[random_index] = arr[i]
        arr[i] = temp
    return arr


def findBestKValue(k_score_values):
    best_k_value = max(k_score_values)
    best_k_value_index = k_score_values.index(best_k_value)
    return "Best value with k = {} with accuracy mean of {:.2%}, standard " \
           "deviation of {:.2} and execution time of {:.4} seconds"\
        .format(best_k_value_index + 1, best_k_value[0], best_k_value[1], best_k_value[2])


def getUltimateList(list1, list2):
    ultimate_list = []
    for i in range(len(list1)):
        ultimate_list.append(list1[i])
        ultimate_list.append(list2[i])
    return ultimate_list


def adjustDictionaries(ultimate_pos, ultimate_neg):
    len1 = len(ultimate_pos[0])
    len2 = len(ultimate_neg[0])
    minimum = min(len1, len2)

    if len1 == minimum:
        for i in range(len(ultimate_neg)):
            ultimate_neg[i] = ultimate_neg[i][0:minimum]
    else:
        for i in range(len(ultimate_pos)):
            ultimate_pos[i] = ultimate_pos[i][0:minimum]


# desired_value: 0 if you want to get the attributes of the image, 1 if you want to get the target
def getSpecificsValues(labeled_data, desired_value):
    specific_values = []
    for i in range(len(labeled_data)):
        specific_values.append(labeled_data[i][desired_value])
    return specific_values

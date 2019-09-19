# author        : Esteban
# course        : CS-691 Data Mining
# name          : utils.py
# date          : 20190920
# python_version: 3.7
# notes         : Assignment2
# description   : Needed for esteban_murillo_assignment2.py to run
# ==============================================================================

import os
import random

import global_variables as global_vars

from itertools import combinations


def partitionDataSet(images_names):
    size_per_fold = len(images_names) / global_vars.x_fold
    partitioned_list = []
    sub_fold = []

    for i in range(len(images_names)):
        sub_fold.append(images_names[i])
        if i % size_per_fold == size_per_fold - 1:
            partitioned_list.append(sub_fold)
            sub_fold = []

    return partitioned_list


def getFoldsCombinations(partitioned_data):
    complete_set = []
    broken_down_set = []
    comb = combinations(partitioned_data, global_vars.x_fold - 1)
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


def assignLabel(_image_folder1, _image_folder2, attributes):
    labeled_data = []
    num_files_folder1 = len([f for f in os.listdir(_image_folder1) if os.path.isfile(os.path.join(_image_folder1, f))])
    for i in range(len(attributes)):
        if i < num_files_folder1:
            labeled_data.append((attributes[i], global_vars.folder1_image_type))
        else:
            labeled_data.append((attributes[i], global_vars.folder2_image_type))
    return labeled_data


# desired_value: 0 if you want to get the attributes of the image, 1 if you want to get the target
def getSpecificsValues(labeled_data, desired_value):
    specific_values = []
    for i in range(len(labeled_data)):
        specific_values.append(labeled_data[i][desired_value])
    return specific_values


def findBestKValue(k_score_values):
    best_k_value = max(k_score_values)
    best_k_value_index = k_score_values.index(best_k_value)
    return "Best value with k = {} with a mean of {}".format(best_k_value_index + 1, best_k_value)


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

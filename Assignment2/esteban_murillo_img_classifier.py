# author        : Esteban
# course        : CS-691 Data Mining
# name          : esteban_murillo_img_classifier.py
# date          : 20190920
# usage         : python3 esteban_murillo_img_classifier.py
# python_version: 3.7
# notes         : Assignment2
# description   :
# ==============================================================================

import os
import random
import numpy as np
from PIL import Image
from itertools import combinations
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

_image_file_extension = ".jpg"
_folder1_image_type = "Insect"
_folder2_image_type = "Ocean"
_image_folder1 = "InsectsTmp/"
_image_folder2 = "OceanTmp/"
_max_num_of_k_tries = 5
_x_fold = 5
_debug = False


def main():
    images_names = loadImages(_image_folder1, _image_folder2)
    rgb_values = getRGBAttributesForAllFiles(images_names)
    labeled_data = shuffle(assignLabel(_image_folder1, _image_folder2, rgb_values))
    k_score_values = kFolding(labeled_data)
    print(k_score_values)
    print(findBestKValue(k_score_values))

    # Code for manual 5-fold begins

    partitioned_data = partitionDataSet(images_names)
    print(partitioned_data)
    print()
    getFoldsCombinations(partitioned_data)


def kFolding(labeled_data):
    X = getSpecificsValues(labeled_data, 0)
    y = getSpecificsValues(labeled_data, 1)
    k_score_values = []

    for k_neighbors in range(1, _max_num_of_k_tries + 1):
        knn_cv = KNeighborsClassifier(k_neighbors)
        cv_scores = cross_val_score(knn_cv, X, y, cv=_x_fold)
        k_score_values.append(np.mean(cv_scores))

    return k_score_values


# desired_value: 0 if you want to get the attributes of the image, 1 if you want to get the target
def getSpecificsValues(labeled_data, desired_value):
    specific_values = []
    for i in range(len(labeled_data)):
        specific_values.append(labeled_data[i][desired_value])
    return specific_values


def getRGBAttributesForAllFiles(images_names):
    rgb_values = []
    for i in range(len(images_names)):
        rgb_values.append(getRGBAttributes(images_names[i]))
    return rgb_values


def getRGBAttributes(file_name):
    try:
        img = Image.open(file_name)
        img_data = img.load()
    except FileNotFoundError:
        return "Could not find image\n"

    # [R, G, B]
    rgb = [0, 0, 0]
    for x in range(img.size[0]):
        for y in range(img.size[1]):
            r, g, b = img_data[x, y]
            rgb[0] += r
            rgb[1] += g
            rgb[2] += b

    return tuple(rgb)


def findBestKValue(k_score_values):
    best_k_value = max(k_score_values)
    best_k_value_index = k_score_values.index(best_k_value)
    return "Best value with k = {} with a mean of {}".format(best_k_value_index + 1, best_k_value)


def partitionDataSet(images_names):
    size_per_fold = len(images_names) / _x_fold
    partitioned_list = []
    sub_fold = []

    for i in range(len(images_names)):
        sub_fold.append(images_names[i])
        if i % size_per_fold == size_per_fold - 1:
            partitioned_list.append(sub_fold)
            sub_fold = []

    return partitioned_list


def getFoldsCombinations(partitioned_data):
    comb = combinations(partitioned_data, 4)
    for i in list(comb):
        print(i)

    for i in range(len(partitioned_data)):
        print("This is the test set:", partitioned_data[-1 - i])


def loadImages(_image_folder1, _image_folder2):
    full_files_names = []
    directory = os.fsencode(_image_folder1)

    for file in os.listdir(directory):
        file_name = os.fsdecode(file)
        if file_name.endswith(_image_file_extension):
            full_files_names.append(_image_folder1 + file_name)

    directory = os.fsencode(_image_folder2)

    for file in os.listdir(directory):
        file_name = os.fsdecode(file)
        if file_name.endswith(_image_file_extension):
            full_files_names.append(_image_folder2 + file_name)

    return full_files_names


def assignLabel(_image_folder1, _image_folder2, attributes):
    labeled_data = []
    num_files_folder1 = len([f for f in os.listdir(_image_folder1) if os.path.isfile(os.path.join(_image_folder1, f))])
    for i in range(len(attributes)):
        if i < num_files_folder1:
            labeled_data.append((attributes[i], _folder1_image_type))
        else:
            labeled_data.append((attributes[i], _folder2_image_type))
    return labeled_data


def shuffle(arr):
    for i in range(len(arr) - 1, -1, -1):
        random_index = random.randint(0, len(arr) - 1)
        temp = arr[random_index]
        arr[random_index] = arr[i]
        arr[i] = temp
    return arr


if __name__ == "__main__":
    main()

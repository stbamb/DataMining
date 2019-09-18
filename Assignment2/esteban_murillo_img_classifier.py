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
from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier


_image_folder1 = "InsectsTmp/"
_image_folder2 = "OceanTmp/"
_folder1_image_type = "Insect"
_folder2_image_type = "Ocean"
_image_file_extension = ".jpg"
_x_fold = 5
_debug = False


def main():
    images_names = loadImages(_image_folder1, _image_folder2)

    rgb_values = []
    for i in range(len(images_names)):
        rgb_values.append(getAvgRGB(images_names[i]))
    #print(rgbValues)

    labeled_data = shuffle(assignLabel(_image_folder1, _image_folder2, rgb_values))
    print(labeled_data)

    kFold(labeled_data)


def kFold(images_names):
    data = np.asarray(images_names)

    k_fold = KFold(_x_fold)

    # enumerate splits
    iteration = 1
    for train, test in k_fold.split(data):
        print("Model:", iteration)
        print('train:\n %s, \ntest:\n %s' % (data[train], data[test]))
        iteration += 1


def partitionDataSet(images_names):
    size_per_fold = len(images_names) / _x_fold
    partitioned_list = []
    sub_fold = []

    for i in range(len(images_names)):
        sub_fold.append(images_names[i])
        if i % size_per_fold == 9:
            partitioned_list.append(sub_fold)
            sub_fold = []

    return partitioned_list


def getAvgRGB(fileName):
    try:
        img = Image.open(fileName)
        img_data = img.load()
    except FileNotFoundError:
        return "Could not find image\n"

    # [R, G, B, TotalPixels]
    rgb = [0, 0, 0]
    for x in range(img.size[0]):
        for y in range(img.size[1]):
            r, g, b = img_data[x, y]
            rgb[0] += r
            rgb[1] += g
            rgb[2] += b
            #rgb[3] += 1

    return rgb


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

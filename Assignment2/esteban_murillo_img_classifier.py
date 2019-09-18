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
from numpy import array
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier


_image_folder1 = "InsectsTmp/"
_image_folder2 = "OceanTmp/"
_folder1_image_type = "Insect"
_folder2_image_type = "Ocean"
_image_file_extension = ".jpg"
_psychonaut = "psychonaut" + _image_file_extension
_x_fold = 3
_debug = False


def main():
    imagesNames = loadImages(_image_folder1, _image_folder2)
    #partitionedList = partitionDataSet(imagesNames)
    print(imagesNames)
    #print(partitionedList)

    rgbValues = []
    for i in range(len(imagesNames)):
        rgbValues.append(getAvgRGB(imagesNames[i]))

    #print(rgbValues)

    #kFold(imagesNames)

    labeledData = shuffle(assignLabel(_image_folder1, _image_folder2, rgbValues))
    print(labeledData)


def kFold(imagesNames):
    data = np.asarray(imagesNames)

    #data = array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])

    kfold = KFold(_x_fold)

    # enumerate splits
    for train, test in kfold.split(data):
        print('train: %s, test: %s' % (data[train], data[test]))


def partitionDataSet(imagesNames):
    sizePerFold = len(imagesNames) / _x_fold
    partitionedList = []
    subFold = []

    for i in range(len(imagesNames)):
        subFold.append(imagesNames[i])
        if i % sizePerFold == 9:
            partitionedList.append(subFold)
            subFold = []

    return partitionedList


def getAvgRGB(fileName):
    try:
        img = Image.open(fileName)
        imgData = img.load()
    except FileNotFoundError:
        return "Could not find image\n"

    # [R, G, B, TotalPixels]
    rgb = [0, 0, 0, 0]
    for x in range(img.size[0]):
        for y in range(img.size[1]):
            r, g, b = imgData[x, y]
            rgb[0] += r
            rgb[1] += g
            rgb[2] += b
            rgb[3] += 1

    return rgb


def loadImages(_image_folder1, _image_folder2):
    fullFilesNames = []
    directory = os.fsencode(_image_folder1)

    for file in os.listdir(directory):
        fileName = os.fsdecode(file)
        if fileName.endswith(_image_file_extension):
            fullFilesNames.append(_image_folder1 + fileName)

    directory = os.fsencode(_image_folder2)

    for file in os.listdir(directory):
        fileName = os.fsdecode(file)
        if fileName.endswith(_image_file_extension):
            fullFilesNames.append(_image_folder2 + fileName)

    return fullFilesNames


def assignLabel(_image_folder1, _image_folder2, attributes):
    labeledData = []
    numFilesFolder1 = len([f for f in os.listdir(_image_folder1) if os.path.isfile(os.path.join(_image_folder1, f))])
    for i in range(len(attributes)):
        if i < numFilesFolder1:
            labeledData.append((attributes[i], _folder1_image_type))
        else:
            labeledData.append((attributes[i], _folder2_image_type))
    return labeledData


def shuffle(arr):
    for i in range(len(arr) - 1, -1, -1):
        random_index = random.randint(0, len(arr) - 1)
        temp = arr[random_index]
        arr[random_index] = arr[i]
        arr[i] = temp
    return arr


if __name__ == "__main__":
    main()

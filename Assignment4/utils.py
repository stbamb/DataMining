# author        : Esteban
# course        : CS-691 Data Mining
# name          : utils.py
# date          : 20191025
# python_version: 3.7
# notes         : Assignment4
# description   : Needed for main.py to run
# ==============================================================================
import os
import re

import config


def gatherFeatures():
    features = []
    files_text = []
    i = 0

    file_names = getFileNames()
    for file in file_names:
        files_text.append(retrieveText(file))

    for file in files_text:
        text_features = [file_names[i], len(file), getNumberComments(file)]
        features.append(text_features)
        i += 1

    return features


def retrieveText(file_name):
    text = []
    try:
        with open(file_name) as f:
            datafile = f.readlines()
            for line in datafile:
                text.append(line)
    except FileNotFoundError:
        print(file_name, " could not be read")
    return text


def getFileNames():
    file_names = []
    directory = os.fsencode(config.SOURCE_CODE_FOLDER)
    try:
        for file in os.listdir(directory):
            file_name = os.fsdecode(file)
            file_names.append(config.SOURCE_CODE_FOLDER + file_name)
    except FileNotFoundError:
        print("Directory not found")
    return file_names


def getNumberComments(file_text):
    num_comments = 0
    for line in file_text:
        num_comments += line.count(config.JAVA_SINGLE_LINE_COMMENT)
    return num_comments

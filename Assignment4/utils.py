# author        : Esteban
# course        : CS-691 Data Mining
# name          : utils.py
# date          : 20191024
# python_version: 3.7
# notes         : Assignment4
# description   : Needed for main.py to run
# ==============================================================================
import os

import config


def gatherFeatures():
    features = []
    files_text = []
    file_names = getFileNames()
    i = 0

    for file in file_names:
        files_text.append(retrieveText(file))

    for file in files_text:
        text_features = [len(file), getNumberOfOccurrences(file)[0], getNumberOfOccurrences(file)[1]]
        features.append((text_features, file_names[i]))
        i += 1

    return features


def retrieveText(file_name):
    text = []
    try:
        with open(config.SOURCE_CODE_FOLDER + file_name) as f:
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
            file_names.append(file_name)
    except FileNotFoundError:
        print("Directory not found")
    return file_names


def getNumberOfOccurrences(file_text):
    num_comments = 0
    num_imports = 0
    for line in file_text:
        num_comments += line.count(config.JAVA_SINGLE_LINE_COMMENT)
        num_imports += line.count(config.JAVA_IMPORT_STATEMENT)
    return num_comments, num_imports

# author        : Esteban
# course        : CS-691 Data Mining
# name          : utils.py
# date          : 20191109
# python_version: 3.7
# notes         : Assignment5
# description   : Needed for main.py to run
# ==============================================================================
import csv
import os

import config


def loadFileNames():
    file_names = []
    directory = os.fsencode(config.SRC_FOLDER)
    try:
        for file in os.listdir(directory):
            file_name = os.fsdecode(file)
            file_names.append(file_name)
    except FileNotFoundError:
        print("Directory not found")
    return file_names


def fileExists(file_name):
    return os.path.exists(file_name)


def getCSVWriteInfo(file_names, info):
    writing_info = info[:]
    header_line = ['Feature' + str(i + 1) for i in range(config.NUMBER_OF_FEATURES_TO_EXTRACT)]
    for i in range(len(writing_info)):
        writing_info[i].insert(0, file_names[i])
    writing_info.insert(0, ['Name'] + header_line)
    return writing_info


def writeToCSV(info):
    with open(config.OUTPUT_CSV_FILE, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerows(info)
    csv_file.close()


def loadCSVInfo():
    rows = []
    with open(config.OUTPUT_CSV_FILE, 'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        for row in csv_reader:
            rows.append(row[1:])
    del rows[0]
    for i in range(len(rows)):
        for j in range(len(rows[0])):
            rows[i][j] = float(rows[i][j])
    return rows


# Not a real function, this was written only because I couldn't figure out what the bug was ¯\_(ツ)_/¯
def fixArray(mfcss):
    for element in mfcss:
        del element[0]
    return mfcss


def createLabeledData(file_names, mfccs):
    labeled_data = []
    for i in range(len(mfccs)):
        labeled_tuple = mfccs[i], file_names[i]
        labeled_data.append(labeled_tuple)
    return labeled_data

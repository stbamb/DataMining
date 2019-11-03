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


def getCSVWriteInfo(info):
    writing_info = info[:]
    header_line = ['Feature' + str(i + 1) for i in range(config.NUMBER_OF_FEATURES_TO_EXTRACT)]
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
            rows.append(row)
    del rows[0]

    for i in range(len(rows)):
        for j in range(len(rows[0])):
            rows[i][j] = float(rows[i][j])

    return rows

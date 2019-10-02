# author        : Esteban
# course        : CS-691 Data Mining
# name          : utils.py
# date          : 20191005
# python_version: 3.7
# notes         : Assignment3
# description   : Needed for esteban_murillo_assignment3.py to run
# ==============================================================================
import os

import global_variables


def getAllReviewsText(reviews_files_names):
    text = ""
    for fileName in reviews_files_names:
        try:
            with open(fileName) as f:
                text += f.read().replace('\n', '')
                f.close()
        except FileNotFoundError:
            print("Error reading", fileName)
        except UnicodeDecodeError:
            print("Invalid character in", fileName)
    return text


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

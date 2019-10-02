# author        : Esteban
# course        : CS-691 Data Mining
# name          : nlp_tools.py
# date          : 20191005
# python_version: 3.7
# notes         : Assignment3
# description   : Needed for esteban_murillo_assignment3.py to run
# ==============================================================================
import heapq
import re

import nltk
from textatistic import Textatistic

import global_variables


def getBookResults(book_data):
    book = Textatistic(book_data[2])
    fres_score = book.flesch_score
    school_level = determineSchoolLevel(fres_score)
    return "Title: {}\nAuthor: {}\nFlesch reading-ease score: {}\nSchool level: {}". \
        format(book_data[0], book_data[1], fres_score, school_level)


def determineSchoolLevel(fres_score):
    school_level_upper_bounds = [30, 50, 60, 70, 80, 90, 100]
    school_levels = ["College graduate", "College", "Tenth to twelfth grade", "Eight to ninth grade",
                     "Seventh grade", "Sixth grade", "Fifth grade"]
    index = -1
    for i in range(len(school_level_upper_bounds)):
        if fres_score < school_level_upper_bounds[i]:
            index = i
            break

    return school_levels[index]


# Code taken from https://www.geeksforgeeks.org/bag-of-words-bow-model-in-nlp/
# and modified to work accordingly
def getBOW(book_content):
    dataset = nltk.sent_tokenize(book_content)
    word2count = {}

    for i in range(len(dataset)):
        dataset[i] = dataset[i].lower()
        dataset[i] = re.sub(r'\W', ' ', dataset[i])
        dataset[i] = re.sub(r'\s+', ' ', dataset[i])

    for data in dataset:
        words = nltk.word_tokenize(data)
        for word in words:
            if word not in word2count.keys():
                word2count[word] = 1
            else:
                word2count[word] += 1

    freq_words = heapq.nlargest(global_variables.max_num_words, word2count, key=word2count.get)
    return word2count

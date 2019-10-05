# author        : Esteban
# course        : CS-691 Data Mining
# name          : nlp_tools.py
# date          : 20191005
# python_version: 3.7
# notes         : Assignment3
# description   : Needed for esteban_murillo_assignment3.py to run
# ==============================================================================
import re

import nltk
from nltk.corpus import stopwords
from textatistic import Textatistic


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


# Code taken from https://www.geeksforgeeks.org/bag-of-words-bow-model-in-nlp/ and modified to work accordingly
def getBOW(documents_text, all_words):
    all_reviews = []
    for document in documents_text:
        filtered_review = cleanupText(document[0])
        word_count = addWords(all_words)
        for data in filtered_review:
            words = nltk.word_tokenize(data)
            for word in words:
                if word not in word_count.keys():
                    word_count[word] = 1
                else:
                    word_count[word] += 1
        all_reviews.append(word_count)
    return all_reviews


def addWords(all_words):
    dictionary = {}
    for word in all_words:
        dictionary[word] = 0
    return dictionary


def getAllWords(documents_text):
    word_count = []
    for document in documents_text:
        filtered_review = cleanupText(document[0])
        for data in filtered_review:
            words = nltk.word_tokenize(data)
            for word in words:
                if word not in word_count:
                    word_count.append(word)
    return word_count


def cleanupText(review):
    dataset = nltk.word_tokenize(review)

    for i in range(len(dataset)):
        dataset[i] = dataset[i].lower()
        dataset[i] = re.sub(r'\W', ' ', dataset[i])
        dataset[i] = re.sub(r'\s+', ' ', dataset[i])
        dataset[i] = re.sub(r'\d+', ' ', dataset[i])

    stop_words = set(stopwords.words('english'))
    stop_words.add("br")
    stop_words.add("n t")
    stop_words.add(" s")
    stop_words.add(" m")
    stop_words.add(" ")
    filtered_review = []

    for word in dataset:
        if word not in stop_words:
            filtered_review.append(word)

    return filtered_review


# Each empty space represent a new word. EZPC ¯\_(ツ)_/¯
def getTotalWordsInDoc(doc):
    counter = 1
    for word in doc:
        if word == " ":
            counter += 1
    return counter


def getNormalizedTermFrequency(reviews, bow):
    freq_table = []
    for i in range(len(reviews)):
        doc_freq = []
        total_words_count = getTotalWordsInDoc(reviews[i][0])
        for appearances in bow[i].values():
            freq = appearances / total_words_count
            doc_freq.append(freq)
        freq_table.append(doc_freq)
    return freq_table

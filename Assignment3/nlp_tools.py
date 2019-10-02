# author        : Esteban
# course        : CS-691 Data Mining
# name          : nlp_tools.py
# date          : 20191005
# python_version: 3.7
# notes         : Assignment3
# description   : Needed for esteban_murillo_assignment3.py to run
# ==============================================================================
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

    print(fres_score)
    return school_levels[index]
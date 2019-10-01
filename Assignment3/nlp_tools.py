# author        : Esteban
# course        : CS-691 Data Mining
# name          : nlp_tools.py
# date          : 20191005
# python_version: 3.7
# notes         : Assignment3
# description   : Needed for esteban_murillo_assignment3.py to run
# ==============================================================================
import enum
import math

from textatistic import Textatistic


def getBookResults(book_data):
    book = Textatistic(book_data[2])
    fres_score = book.flesch_score
    school_level = determineSchoolLevel(fres_score)
    return "Title: {}\nAuthor: {}\nFlesch reading-ease score: {}\nSchool level: {}". \
        format(book_data[0], book_data[1], fres_score, school_level)


def determineSchoolLevel(fres_score):
    fres_score = roundup(fres_score)
    for level in SchoolLevel:
        if fres_score == level.value:
            return level.name


def roundup(x):
    num = int(math.ceil(x / 10.0)) * 10
    if num <= 20:
        num = 30
    elif num <= 40:
        num = 50
    elif num > 100:
        num = 100
    return num


class SchoolLevel(enum.Enum):
    COLLEGE_GRADUATE = 30
    COLLEGE = 50
    TENTH_TO_TWELFTH_GRADE = 60
    EIGHTH_TO_NINTH_GRADE = 70
    SEVENTH_GRADE = 80
    SIXTH_GRADE = 90
    FIFTH_GRADE = 100

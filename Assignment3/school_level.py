# author        : Esteban
# course        : CS-691 Data Mining
# name          : school_level.py
# date          : 20191005
# usage         : python3 esteban_murillo_assignment3.py
# python_version: 3.7
# notes         : Assignment3
# description   :
# ==============================================================================
import enum


class SchoolLevel(enum.Enum):
    COLLEGE_GRADUATE = 30
    COLLEGE = 50
    TENTH_TO_TWELFTH_GRADE = 60
    EIGHTH_TO_NINTH_GRADE = 70
    SEVENTH_GRADE = 80
    SIXTH_GRADE = 90
    FIFTH_GRADE = 100

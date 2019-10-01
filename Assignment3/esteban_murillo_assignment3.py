# author        : Esteban
# course        : CS-691 Data Mining
# name          : esteban_murillo_assignment3.py
# date          : 20191005
# usage         : python3 esteban_murillo_assignment3.py
# python_version: 3.7
# notes         : Assignment3
# description   :
# ==============================================================================
import global_variables
import nlp_tools
import pdf_tools


def main():
    book_data = pdf_tools.extractInformation(global_variables.book_name1)
    book_output = nlp_tools.getBookResults(book_data)
    print(book_output)


if __name__ == '__main__':
    main()

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
import utils


def main():
    # Minimal performance
    # book_data = pdf_tools.extractInformation(global_variables.book_name1)
    # book_output = nlp_tools.getBookResults(book_data)
    # print(book_output)

    review_files_names = utils.getReviewFilesNames(global_variables.pos_reviews_folder,
                                                   global_variables.neg_reviews_folder)

    positive_reviews_text = utils.getAllReviewsText(review_files_names[0])
    negative_reviews_text = utils.getAllReviewsText(review_files_names[1])

    positive_bow = nlp_tools.getBOW(positive_reviews_text)
    negative_bow = nlp_tools.getBOW(negative_reviews_text)

    if global_variables.verbose:
        print("POSITIVE REVIEWS")
        print(positive_bow)
        print("NEGATIVE REVIEWS")
        print(negative_bow)


if __name__ == '__main__':
    main()

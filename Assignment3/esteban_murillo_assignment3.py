# author        : Esteban
# course        : CS-691 Data Mining
# name          : esteban_murillo_assignment3.py
# date          : 20191005
# usage         : python3 esteban_murillo_assignment3.py
# python_version: 3.7
# notes         : Assignment3
# description   :
# ==============================================================================
import time

import numpy as np
from sklearn import metrics
from sklearn.model_selection import cross_val_score, KFold
from sklearn.naive_bayes import GaussianNB

import global_variables
import inhouse_knn
import nlp_tools
import pdf_tools
import utils
from sklearn.neighbors import KNeighborsClassifier


def main():
    # Minimal performance
    book_data = pdf_tools.extractInformation(global_variables.book_name2)
    book_output = nlp_tools.getBookResults(book_data)
    print(book_output)

    review_files_names = utils.getReviewFilesNames(global_variables.pos_reviews_folder,
                                                   global_variables.neg_reviews_folder)

    positive_reviews_text = utils.getAllReviewsText(review_files_names[0])
    negative_reviews_text = utils.getAllReviewsText(review_files_names[1])

    all_pos_bow = nlp_tools.getAllWords(positive_reviews_text)
    all_neg_bow = nlp_tools.getAllWords(negative_reviews_text)

    positive_bow = nlp_tools.getBOW(positive_reviews_text, all_pos_bow)
    negative_bow = nlp_tools.getBOW(negative_reviews_text, all_neg_bow)

    pos_norm_term_freq_table = nlp_tools.getNormalizedTermFrequency(positive_reviews_text, positive_bow)

    neg_norm_term_freq_table = nlp_tools.getNormalizedTermFrequency(negative_reviews_text, negative_bow)

    utils.adjustDictionaries(pos_norm_term_freq_table, neg_norm_term_freq_table)

    # Add 1 to positive reviews, -1 to negative
    ultimate_pos = utils.addLabels(pos_norm_term_freq_table, 1)
    ultimate_neg = utils.addLabels(neg_norm_term_freq_table, 0)
    ultimate_list = utils.shuffle(ultimate_pos + ultimate_neg)

    # Cross-Fold validation + KNeighborsClassifier (all implemented from scratch)
    partitioned_data = utils.partitionDataSet(ultimate_list)
    complete_set = utils.getFoldsCombinations(partitioned_data)
    score_values = inhouseKNN(complete_set)
    print("\nValues for cross-fold validation + KNeighborsClassifier (all implemented from scratch)")
    print(utils.findBestKValue(score_values))

    # Cross-Fold validation + KNeighborsClassifier (all from sklearn)
    print("\nValues for cross-fold validation + KNeighborsClassifier (all from sklearn)")
    score_values = builtinKNN(ultimate_list)
    print(utils.findBestKValue(score_values))

    # Cross-Fold validation + SVC (all from sklearn)
    print("\nValues for cross-fold validation + Naive Bayes Classifier (all from sklearn)")
    score_values = naiveBayesClassifier(ultimate_list)
    print("Accuracy mean of {:.2%}, standard deviation of {:.2} and execution time of {:.4} seconds".
          format(score_values[0], score_values[1], score_values[2]))


def naiveBayesClassifier(labeled_data):
    X = utils.getSpecificsValues(labeled_data, 0)
    y = utils.getSpecificsValues(labeled_data, 1)

    print(y)

    start_time = time.time()

    nbc = GaussianNB()

    nbc.fit(X, y)

    expected = y
    predicted = nbc.predict(X)

    print(metrics.classification_report(expected, predicted))

    k_fold = KFold(len(y), shuffle=False, random_state=0)

    cv_scores = cross_val_score(nbc, X, y, cv=k_fold)
    total_execution_time = time.time() - start_time

    return np.mean(cv_scores), np.std(cv_scores), total_execution_time


def builtinKNN(labeled_data):
    X = utils.getSpecificsValues(labeled_data, 0)
    y = utils.getSpecificsValues(labeled_data, 1)
    builtin_KNN_k_score_values = []

    for k_neighbors in range(1, global_variables.max_num_of_k_tries + 1):
        start_time = time.time()
        knn_cv = KNeighborsClassifier(k_neighbors)
        cv_scores = cross_val_score(knn_cv, X, y, cv=global_variables.x_fold)
        total_execution_time = time.time() - start_time
        builtin_KNN_k_score_values.append((np.mean(cv_scores), np.std(cv_scores), total_execution_time))

    return builtin_KNN_k_score_values


def inhouseKNN(complete_set):
    iterations = len(complete_set) // 2
    inhouse_KNN_k_score_values = []

    for k in range(1, global_variables.max_num_of_k_tries + 1):
        total_right_predictions = 0
        total_guesses = 0
        fold_accuracies = []
        start_time = time.time()
        for i in range(iterations):
            train_set = complete_set[i]
            test_set = complete_set[-1 - i]
            train_set = utils.joinTrainSet(train_set)
            distances = inhouse_knn.calculateDistance(global_variables.default_distance_algorithm, train_set, test_set)
            right_predictions = inhouse_knn.kNN(distances, k)
            total_right_predictions += right_predictions
            total_guesses += len(test_set)
            accuracy_for_current_fold = right_predictions / len(test_set)
            fold_accuracies.append(accuracy_for_current_fold)

            #if global_variables.verbose:
                #print("train_set {}\n \ntest_set {}\n".format(train_set, test_set))

        total_execution_time = time.time() - start_time
        accuracy = total_right_predictions / total_guesses
        inhouse_KNN_k_score_values.append((accuracy, np.std(fold_accuracies), total_execution_time))

        if global_variables.verbose:
            print("\nRight predictions {} out of {}. Total accuracy {}% with k = {}"
                  .format(total_right_predictions, total_guesses, accuracy, k))

    return inhouse_KNN_k_score_values


if __name__ == '__main__':
    main()

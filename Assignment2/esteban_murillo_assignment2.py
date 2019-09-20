# author        : Esteban
# course        : CS-691 Data Mining
# name          : esteban_murillo_assignment2.py
# date          : 20190920
# usage         : python3 esteban_murillo_assignment2.py
# python_version: 3.7
# notes         : Assignment2
# description   :
# ==============================================================================

import numpy as np
import time

import utils as utils
import image_tools as image_tools
import inhouse_knn as inhouse_knn
import global_variables as global_vars

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score


def main():
    # Load images, extract attributes, assign labels (to know which category the image belongs to)
    images_names = image_tools.loadImages(global_vars.image_folder1, global_vars.image_folder2)
    rgb_values = image_tools.getRGBAttributesForAllFiles(images_names)
    labeled_data = utils.shuffle(utils.assignLabel(global_vars.image_folder1, global_vars.image_folder2, rgb_values))
    print("Currently testing with values of k ranging from 1 to", global_vars.max_num_of_k_tries)
    # NOTE: The same shuffled data is passed to builtinKNN, inhouseKNN, randomForest & SVC

    # Cross-Fold validation + KNeighborsClassifier (all from sklearn)
    print("\nValues for cross-fold validation + KNeighborsClassifier (all from sklearn)")
    score_values = builtinKNN(labeled_data)
    print(utils.findBestKValue(score_values))

    # Cross-Fold validation + KNeighborsClassifier (all implemented from scratch)
    print("\nValues for cross-fold validation + KNeighborsClassifier (all implemented from scratch)")
    partitioned_data = utils.partitionDataSet(labeled_data)
    complete_set = utils.getFoldsCombinations(partitioned_data)
    score_values = inhouseKNN(complete_set)
    print(utils.findBestKValue(score_values))

    # Cross-Fold validation + RandomForestClassifier (all from sklearn)
    print("\nValues for cross-fold validation + RandomForestClassifier (all from sklearn)")
    score_values = randomForestClassifier(labeled_data)
    print(global_vars.extra_algorithms_report_message.format(score_values[0], score_values[1], score_values[2]))

    # Cross-Fold validation + SVC (all from sklearn)
    print("\nValues for cross-fold validation + SVC (all from sklearn)")
    score_values = SVClassifier(labeled_data)
    print(global_vars.extra_algorithms_report_message.format(score_values[0], score_values[1], score_values[2]))


def builtinKNN(labeled_data):
    X = utils.getSpecificsValues(labeled_data, 0)
    y = utils.getSpecificsValues(labeled_data, 1)
    builtin_KNN_k_score_values = []

    for k_neighbors in range(1, global_vars.max_num_of_k_tries + 1):
        start_time = time.time()
        knn_cv = KNeighborsClassifier(k_neighbors)
        cv_scores = cross_val_score(knn_cv, X, y, cv=global_vars.x_fold)
        total_execution_time = time.time() - start_time
        builtin_KNN_k_score_values.append((np.mean(cv_scores), np.std(cv_scores), total_execution_time))

    return builtin_KNN_k_score_values


def inhouseKNN(complete_set):
    iterations = len(complete_set) // 2
    inhouse_KNN_k_score_values = []

    for k in range(1, global_vars.max_num_of_k_tries + 1):
        total_right_predictions = 0
        total_guesses = 0
        fold_accuracies = []
        start_time = time.time()
        for i in range(iterations):
            train_set = complete_set[i]
            test_set = complete_set[-1 - i]
            train_set = utils.joinTrainSet(train_set)
            distances = inhouse_knn.calculateDistance(global_vars.default_distance_algorithm, train_set, test_set)
            right_predictions = inhouse_knn.kNN(distances, k)
            total_right_predictions += right_predictions
            total_guesses += len(test_set)
            accuracy_for_current_fold = right_predictions / len(test_set)
            fold_accuracies.append(accuracy_for_current_fold)

            if global_vars.debug:
                print("train_set {}\n \ntest_set {}\n".format(train_set, test_set))

        total_execution_time = time.time() - start_time
        accuracy = total_right_predictions / total_guesses
        inhouse_KNN_k_score_values.append((accuracy, np.std(fold_accuracies), total_execution_time))

        if global_vars.debug:
            print("Right predictions {} out of {}. Total accuracy {}% with k = {}"
                  .format(total_right_predictions, total_guesses, accuracy, k))

    return inhouse_KNN_k_score_values


def randomForestClassifier(labeled_data):
    start_time = time.time()
    X = utils.getSpecificsValues(labeled_data, 0)
    y = utils.getSpecificsValues(labeled_data, 1)
    cv_scores = cross_val_score(RandomForestClassifier(n_estimators=100), X, y, cv=global_vars.x_fold)
    total_execution_time = time.time() - start_time
    return np.mean(cv_scores), np.std(cv_scores), total_execution_time


def SVClassifier(labeled_data):
    start_time = time.time()
    X = utils.getSpecificsValues(labeled_data, 0)
    y = utils.getSpecificsValues(labeled_data, 1)
    cv_scores = cross_val_score(SVC(gamma='scale'), X, y, cv=global_vars.x_fold)
    total_execution_time = time.time() - start_time
    return np.mean(cv_scores), np.std(cv_scores), total_execution_time


if __name__ == "__main__":
    main()

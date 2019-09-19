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
import utils as utils

import image_tools as image_tools
import inhouse_knn as inhouse_knn
import global_variables as global_vars

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score


def main():
    # Load images, extract attributes, assign labels (to know which category the image belongs to)
    images_names = image_tools.loadImages(global_vars.image_folder1, global_vars.image_folder2)
    rgb_values = image_tools.getRGBAttributesForAllFiles(images_names)
    labeled_data = utils.shuffle(utils.assignLabel(global_vars.image_folder1, global_vars.image_folder2, rgb_values))
    print("Currently testing with values of k ranging from 1 to", global_vars.max_num_of_k_tries)
    # NOTE: The same shuffled data is passed to both builtinKNN & inhouseKNN

    # Cross-Fold validation + KNeighborsClassifier (all from sklearn)
    print("\nValues for cross-Fold validation + KNeighborsClassifier (all from sklearn)")
    builtin_KNN_k_score_values = builtinKNN(labeled_data)
    print(builtin_KNN_k_score_values)
    print(utils.findBestKValue(builtin_KNN_k_score_values))

    # Cross-Fold validation + KNeighborsClassifier (all implemented from scratch)
    print("\nValues for cross-Fold validation + KNeighborsClassifier (all implemented from scratch)")
    partitioned_data = utils.partitionDataSet(labeled_data)
    complete_set = utils.getFoldsCombinations(partitioned_data)
    inhouse_KNN_k_score_values = inhouseKNN(complete_set)
    print(inhouse_KNN_k_score_values)
    print(utils.findBestKValue(inhouse_KNN_k_score_values))


def inhouseKNN(complete_set):
    iterations = len(complete_set) // 2
    inhouse_KNN_k_score_values = []

    for k in range(1, global_vars.max_num_of_k_tries + 1):
        right_predictions = 0
        total_guesses = 0
        for i in range(iterations):
            train_set = complete_set[i]
            test_set = complete_set[-1 - i]
            train_set = utils.joinTrainSet(train_set)
            distances = inhouse_knn.calculateDistance(global_vars.default_distance_algorithm, train_set, test_set)
            right_predictions += inhouse_knn.kNN(distances, k)
            total_guesses += len(test_set)

            if global_vars.debug:
                print("train_set {}\n \ntest_set {}\n".format(train_set, test_set))

        accuracy = right_predictions / total_guesses
        inhouse_KNN_k_score_values.append(accuracy)

        if global_vars.debug:
            print("Right predictions {} out of {}. Total accuracy {}% with k = {}"
                  .format(right_predictions, total_guesses, accuracy, k))

    return inhouse_KNN_k_score_values


def builtinKNN(labeled_data):
    X = utils.getSpecificsValues(labeled_data, 0)
    y = utils.getSpecificsValues(labeled_data, 1)
    builtin_KNN_k_score_values = []

    for k_neighbors in range(1, global_vars.max_num_of_k_tries + 1):
        knn_cv = KNeighborsClassifier(k_neighbors)
        cv_scores = cross_val_score(knn_cv, X, y, cv=global_vars.x_fold)
        builtin_KNN_k_score_values.append(np.mean(cv_scores))

    return builtin_KNN_k_score_values


if __name__ == "__main__":
    main()

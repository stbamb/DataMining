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
    images_names = image_tools.loadImages(global_vars.image_folder1, global_vars.image_folder2)
    rgb_values = image_tools.getRGBAttributesForAllFiles(images_names)
    labeled_data = utils.shuffle(utils.assignLabel(global_vars.image_folder1, global_vars.image_folder2, rgb_values))
    k_score_values = builtinKNN(labeled_data)
    print(k_score_values)
    print(utils.findBestKValue(k_score_values))

    # Code for manual 5-fold begins

    partitioned_data = utils.partitionDataSet(labeled_data)
    print(partitioned_data)
    print()
    complete_set = utils.getFoldsCombinations(partitioned_data)
    inhouseKNN(complete_set)


def inhouseKNN(complete_set):
    default_distance_algorithm = inhouse_knn.manhattanDistance

    for i in range((len(complete_set) // 2)):
        train_set = complete_set[i]
        test_set = complete_set[-1 - i]
        train_set = utils.joinTrainSet(train_set)

        print("train_set {}\n \ntest_set {}\n".format(train_set, test_set))
        distances = inhouse_knn.calculateDistance(default_distance_algorithm, train_set, test_set)
        print(distances)
        print()


def builtinKNN(labeled_data):
    X = utils.getSpecificsValues(labeled_data, 0)
    y = utils.getSpecificsValues(labeled_data, 1)
    k_score_values = []

    for k_neighbors in range(1, global_vars.max_num_of_k_tries + 1):
        knn_cv = KNeighborsClassifier(k_neighbors)
        cv_scores = cross_val_score(knn_cv, X, y, cv=global_vars.x_fold)
        k_score_values.append(np.mean(cv_scores))

    return k_score_values


if __name__ == "__main__":
    main()

# author        : Esteban
# course        : CS-691 Data Mining
# name          : main.py
# date          : 20191025
# usage         : python3 main.py
# python_version: 3.7
# notes         : Assignment4
# description   :
# ==============================================================================
from sklearn.cluster import KMeans

import clustering
import config
import utils


def main():
    labeled_features = utils.gatherFeatures()
    custom_clusters = clustering.k_means_clustering(labeled_features, config.NUMBER_OF_CLUSTERS)
    sklearn_clusters = sklearnClustering(labeled_features)
    printResults(custom_clusters, sklearn_clusters)

    if custom_clusters == sklearn_clusters:
        print("Same elements were clustered together for both algorithms")


def printResults(custom_clusters, sklearn_clusters):
    print("*" * 128, "CUSTOM K-MEANS ALGORITHM", "*" * 128)
    for i in range(config.NUMBER_OF_CLUSTERS):
        programs = [element[1] for element in custom_clusters[i]]
        print("Elements in cluster {}:\n\n{}\n\nWith a total of {} elements\n".format(i + 1, programs, len(programs)))

    print("*" * 128, "SKLEARN K-MEANS ALGORITHM", "*" * 128)
    for i in range(config.NUMBER_OF_CLUSTERS):
        programs = [element[1] for element in sklearn_clusters[i]]
        print("Elements in cluster {}:\n\n{}\n\nWith a total of {} elements\n".format(i + 1, programs, len(programs)))


def sklearnClustering(labeled_features):
    clusters = getClusterValues(config.NUMBER_OF_CLUSTERS)
    features = [feature[0] for feature in labeled_features]
    km = KMeans(n_clusters=2, init='random', n_init=1, max_iter=config.MAX_NUMBER_OF_ITERATIONS)
    y_km = km.fit_predict(features)

    pos = 0
    for value in y_km:
        clusters[value].append(labeled_features[pos])
        pos += 1

    if config.VERBOSE:
        print(km.cluster_centers_)
        print(km.inertia_)

    return clusters


def getClusterValues(k):
    sklearn_clusters = []
    for i in range(k):
        sklearn_clusters.append([])
    return sklearn_clusters


if __name__ == "__main__":
    main()

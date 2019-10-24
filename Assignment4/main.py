# author        : Esteban
# course        : CS-691 Data Mining
# name          : main.py
# date          : 20191024
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
    custom_clustering_data = clustering.k_means_clustering(labeled_features, config.NUMBER_OF_CLUSTERS)
    sklearn_clusters = sklearnClustering(labeled_features, config.NUMBER_OF_CLUSTERS)
    printResults(custom_clustering_data, sklearn_clusters, config.NUMBER_OF_CLUSTERS)
    scores = compareKs(labeled_features)

    print("*" * 64, "EXPLORING THE BEST K VALUE FOR K-MEANS", "*" * 64)
    print("*" * 69, "SELF-IMPLEMENTED VS. SKLEARN", "*" * 69)

    custom_score = [float("%.2f" % sum(x)) for x in scores[0]]
    sklearn_score = [float("%.2f" % x) for x in scores[1]]

    print("SCORES FOR CUSTOM K-MEANS ALGORITHM:  {}".format(custom_score))
    print("SCORES FOR SKLEARN K-MEANS ALGORITHM: {}".format(sklearn_score))


def compareKs(labeled_features):
    custom_clustering_scores = []
    sklearn_clustering_scores = []
    for k in range(1, config.MAX_NUMBER_OF_ITERATIONS + 1):
        custom_clustering_data = clustering.k_means_clustering(labeled_features, k)
        sklearn_clusters_data = sklearnClustering(labeled_features, k)
        custom_clustering_scores.append(custom_clustering_data[1])
        sklearn_clustering_scores.append(sklearn_clusters_data[1])
        if config.DEBUG:
            printResults(custom_clustering_data, sklearn_clusters_data, k)
    return custom_clustering_scores, sklearn_clustering_scores


def printResults(custom_clusters, sklearn_clusters, k):
    print("*" * 64, "CUSTOM K-MEANS ALGORITHM WITH K =", k, "*" * 64)
    for i in range(k):
        programs = [element[1] for element in custom_clusters[0][i]]
        print("Cluster {} (with {} elements and SSE of {:.2f}):\n\n{}\n"
              .format(i + 1, len(programs), custom_clusters[1][i], programs))

    print("*" * 64, "SKLEARN K-MEANS ALGORITHM WITH K =", k, "*" * 64)
    for i in range(k):
        programs = [element[1] for element in sklearn_clusters[0][i]]
        print("Cluster {} (with {} elements):\n\n{}\n"
              .format(i + 1, len(programs), programs))


def sklearnClustering(labeled_features, k):
    clusters = getClusterValues(k)
    features = [feature[0] for feature in labeled_features]
    km = KMeans(n_clusters=k, init='random', n_init=1, max_iter=config.MAX_NUMBER_OF_ITERATIONS)
    y_km = km.fit_predict(features)

    pos = 0
    for value in y_km:
        clusters[value].append(labeled_features[pos])
        pos += 1

    return clusters, km.inertia_


def getClusterValues(k):
    sklearn_clusters = []
    for i in range(k):
        sklearn_clusters.append([])
    return sklearn_clusters


if __name__ == "__main__":
    main()

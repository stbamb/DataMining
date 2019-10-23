# author        : Esteban
# course        : CS-691 Data Mining
# name          : clustering.py
# date          : 20191025
# python_version: 3.7
# notes         : Assignment4
# description   : Needed for main.py to run
# ==============================================================================
import math
import random

import config


def k_means_clustering(labeled_features, k):
    features = [feature[0] for feature in labeled_features]
    stop_conditions = []
    new_clusters = []
    iteration = 1

    while continueClustering(stop_conditions):
        if iteration == 1:  # if it is the first time, then pick random data points as centroids
            old_centroids = getRandomCentroids(features, k)
        else:  # otherwise compute averages and make it your new centroids
            old_centroids = new_centroids

        distances = calculateDistance(euclideanDistance, features, old_centroids)

        distances_to_clusters = getDistancesToClusters(distances)
        old_clusters = new_clusters
        new_clusters = assignToCluster(labeled_features, distances_to_clusters)
        new_centroids = getAvgCentroid(new_clusters)

        squared_metrics_sum = getSquaredMetricsSum(distances)

        stop_conditions = getStopConditions(old_clusters, new_clusters, old_centroids, new_centroids, iteration)

        if config.VERBOSE:
            print("*" * 128, "ITERATION #", iteration, "*" * 128)
            print("Labeled features:\n", labeled_features, "\n")
            print("Centroids:\n", old_centroids, "\n")
            print("Distances to clusters:\n", distances_to_clusters, "\n")
            print("Distances:\n", distances, "\n")
            for i in range(len(new_clusters)):
                print("Cluster {}, with {} elements:\n {}\n".format(i + 1, len(new_clusters[i]), new_clusters[i]))
                print("Squared metrics sum {}".format(squared_metrics_sum[i]))
            print("Clusters changed? {}\nCentroids changed? {}\nContinue? {}\nIteration: {} out of {}"
                  .format(stop_conditions[0], stop_conditions[1], stop_conditions[2],
                          iteration, config.MAX_NUMBER_OF_ITERATIONS))

        iteration += 1
    return new_clusters


def continueClustering(conditions):
    continue_clustering = True
    for condition in conditions:
        continue_clustering = continue_clustering and condition
    return continue_clustering


def calculateDistance(default_distance_algorithm, *training_data_set, **test_data_set):
    return default_distance_algorithm(*training_data_set, **test_data_set)


def manhattanDistance(training_data_set, test_data_set):
    manhattan_distances = []
    for test_set in test_data_set:
        distances = []
        for training_set in training_data_set:
            distances.append(sum([(abs(x - y)) for x, y in zip(test_set, training_set)]))
        manhattan_distances.append(distances)
    return manhattan_distances


def euclideanDistance(training_data_set, test_data_set):
    euclidean_distances = []
    for test_set in test_data_set:
        distances = []
        for training_set in training_data_set:
            distances.append(math.sqrt(sum([(x - y) ** 2 for x, y in zip(test_set, training_set)])))
        euclidean_distances.append(distances)
    return euclidean_distances


def getRandomCentroids(features, k):
    centroids = []
    for i in range(k):
        random_centroid_idx = random.randint(0, len(features) - 1)
        while features[random_centroid_idx] in centroids:
            random_centroid_idx = random.randint(0, len(features) - 1)
        centroids.append(features[random_centroid_idx])
    return centroids


def getAvgCentroid(clusters):
    averages = []
    for cluster in clusters:
        features = [element[0] for element in cluster]
        avg_for_cluster = [sum(value) / len(features) for value in zip(*features)]
        averages.append(avg_for_cluster)
    return averages


def getSquaredMetricsSum(distances):
    squared_metrics_sum = [sum(distance) ** 2 for distance in distances]
    return squared_metrics_sum


def getDistancesToClusters(distances):
    distances_to_clusters = []
    for j in range(len(distances[0])):
        dists = []
        for i in range(len(distances)):
            dists.append(distances[i][j])
        distances_to_clusters.append(dists)
    return distances_to_clusters


def assignToCluster(labeled_features, distances_to_clusters):
    clusters = []
    i = 0
    for _ in distances_to_clusters[0]:
        clusters.append([])
    for distance in distances_to_clusters:
        idx = distance.index(min(distance))
        clusters[idx].append(labeled_features[i])
        i += 1
    return clusters


def getStopConditions(old_clusters, new_clusters, old_centroids, new_centroids, iteration):
    continue_clustering = True
    if iteration >= config.MAX_NUMBER_OF_ITERATIONS:
        continue_clustering = not continue_clustering
    return old_clusters != new_clusters, old_centroids != new_centroids, continue_clustering

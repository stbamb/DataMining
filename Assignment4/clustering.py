# author        : Esteban
# course        : CS-691 Data Mining
# name          : clustering.py
# date          : 20191025
# python_version: 3.7
# notes         : Assignment4
# description   : Needed for main.py to run
# ==============================================================================
import random

import config


def k_means_clustering(labeled_features, k):
    features = [feature[0] for feature in labeled_features]
    stop_conditions = []
    new_clusters = []
    iteration = 0

    while continueClustering(stop_conditions, iteration):
        if iteration == 0:  # if it is the first time, then pick random datapoints as centroids
            old_centroids = getRandomCentroids(features, k)
        else:  # otherwise compute averages and make it your new centroids
            old_centroids = new_centroids

        distances = manhattanDistance(features, old_centroids)
        distances_to_clusters = getDistancesToClusters(distances)
        old_clusters = new_clusters
        new_clusters = assignToCluster(labeled_features, distances_to_clusters)
        new_centroids = getAvgCentroid(new_clusters)

        stop_conditions = (old_clusters != new_clusters, old_centroids != new_centroids)
        iteration += 1

        if config.VERBOSE:
            print("*"*128, "ITERATION #", iteration, "*"*128)
            print("Labeled features:\n", labeled_features, "\n")
            print("Centroids:\n", old_centroids, "\n")
            print("Distances to clusters:\n", distances_to_clusters, "\n")
            for i in range(len(new_clusters)):
                print("Cluster {}, with {} elements:\n {}\n".format(i + 1, len(new_clusters[i]), new_clusters[i]))
            print("Clusters changed? {}\nCentroids changed? {}\nIteration: {}\n"
                  .format(stop_conditions[0], stop_conditions[1], iteration))

    return new_clusters


def continueClustering(conditions, iteration):
    flag = True
    if iteration >= config.MAX_NUMBER_OF_ITERATIONS:
        return not flag
    for condition in conditions:
        flag = flag and condition
    return flag


def manhattanDistance(training_data_set, test_data_set):
    manhattan_distances = []
    for i in range(len(test_data_set)):
        A = test_data_set[i]
        distances = []
        for j in range(len(training_data_set)):
            B = training_data_set[j]
            dist = 0
            for k in range(len(A) - 1):
                dist += abs(A[k] - B[k])
            distances.append(dist)
        manhattan_distances.append(distances)
    return manhattan_distances


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

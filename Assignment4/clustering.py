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
    stop_conditions = []
    clusters = []
    iteration = 0

    while continueClustering(stop_conditions, iteration):
        features = [feature[0] for feature in labeled_features]
        centroids = getCentroids(clusters, features, k, iteration)
        distances = manhattanDistance(features, centroids)
        distances_to_clusters = getDistancesToClusters(distances)
        clusters = assignToCluster(labeled_features, distances_to_clusters)
        new_centroids = getAvgCentroid(clusters)
        centroids_changed = didCentroidsChange(centroids, new_centroids)
        stop_conditions.clear()
        stop_conditions.append(centroids_changed)
        iteration += 1

        if config.VERBOSE:
            print("*"*128, "ITERATION #", iteration, "*"*128)
            print("Labeled features:\n", labeled_features, "\n")
            print("Centroids:\n", centroids, "\n")
            print("Distances to clusters:\n", distances_to_clusters, "\n")
            for i in range(len(clusters)):
                print("Cluster {}, with {} elements:\n {}\n".format(i + 1, len(clusters[i]), clusters[i]))
            print("Centroids changed?\n", centroids_changed, "\n")

    return clusters


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


def getCentroids(clusters, features, k, iteration):
    centroids = []
    if iteration == 0:  # if it is the first time, then pick a datapoint at random
        for i in range(k):
            random_centroid_idx = random.randint(0, len(features) - 1)
            while features[random_centroid_idx] in centroids:
                random_centroid_idx = random.randint(0, len(features) - 1)
            centroids.append(features[random_centroid_idx])
    else:  # otherwise, calculate the avg
        centroids = getAvgCentroid(clusters)
    return centroids


def getAvgCentroid(clusters):
    avgs = []
    for cluster in clusters:
        features = [element[0] for element in cluster]
        avg_for_cluster = [sum(value) / len(features) for value in zip(*features)]
        avgs.append(avg_for_cluster)
    return avgs


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


def didCentroidsChange(old_centroids, new_centroids):
    return old_centroids != new_centroids


def didClustersChange(old_clusters, new_clusters):
    return old_clusters != new_clusters

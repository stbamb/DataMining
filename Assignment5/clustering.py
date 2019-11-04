# author        : Esteban
# course        : CS-691 Data Mining
# name          : clustering.py
# date          : 20191109
# python_version: 3.7
# notes         : Assignment5
# description   : Needed for main.py to run
# ==============================================================================
import math
import random

from sklearn.cluster import KMeans, AgglomerativeClustering

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

        distances_to_centroids = getDistancesToCentroids(distances)
        old_clusters = new_clusters
        new_clusters = assignToCluster(labeled_features, distances_to_centroids)

        new_centroids = getAvgCentroid(new_clusters)

        sse = getSSE(new_clusters, new_centroids)

        stop_conditions = getStopConditions(old_clusters, new_clusters, old_centroids, new_centroids, iteration)

        if config.DEBUG:
            print("*" * 64, "ITERATION #", iteration, "*" * 64)
            print("Labeled features:\n", labeled_features, "\n")
            print("Centroids:\n", old_centroids, "\n")
            print("Distances to clusters:\n", distances_to_centroids, "\n")
            print("Distances:\n", distances, "\n")
            for i in range(len(new_clusters)):
                print("Cluster {}, with {} elements and SSE of {:.2}:\n\n{}\n"
                      .format(i + 1, len(new_clusters[i]), sse[i], new_clusters[i]))
            print("Clusters changed? {}\nCentroids changed? {}\nContinue? {}\nIteration: {} out of {}\n"
                  .format(stop_conditions[0], stop_conditions[1], stop_conditions[2],
                          iteration, config.MAX_NUMBER_OF_ITERATIONS))

        iteration += 1
    return new_clusters  # , sse


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


def getSSE(clusters, centroids):
    sse = []
    distance_to_centroids = []
    for cluster in clusters:
        distance_to_centroids.append([dist[0] for dist in cluster])
    for i in range(len(distance_to_centroids)):
        distances = euclideanDistance(distance_to_centroids[i], centroids)
        distances = distances[i]
        sse.append(sum([value ** 2 for value in distances]))
    return sse


def getDistancesToCentroids(distances):
    distances_to_centroids = []
    for j in range(len(distances[0])):
        dists = []
        for i in range(len(distances)):
            dists.append(distances[i][j])
        distances_to_centroids.append(dists)
    return distances_to_centroids


def assignToCluster(labeled_features, distances_to_centroids):
    clusters = []
    i = 0
    for _ in distances_to_centroids[0]:
        clusters.append([])
    for distance in distances_to_centroids:
        idx = distance.index(min(distance))
        clusters[idx].append(labeled_features[i])
        i += 1
    return clusters


def getStopConditions(old_clusters, new_clusters, old_centroids, new_centroids, iteration):
    continue_clustering = True
    if iteration >= config.MAX_NUMBER_OF_ITERATIONS:
        continue_clustering = not continue_clustering
    return old_clusters != new_clusters, old_centroids != new_centroids, continue_clustering


def sklearnKMeansClustering(labeled_features, k):
    clusters = getClusterValues(k)
    features = [feature[0] for feature in labeled_features]
    km = KMeans(n_clusters=k, init='random', n_init=1, max_iter=config.MAX_NUMBER_OF_ITERATIONS)
    y_km = km.fit_predict(features)
    pos = 0
    for value in y_km:
        clusters[value].append(labeled_features[pos])
        pos += 1
    return clusters


def sklearnAgglomerativeClustering(labeled_features, k):
    clusters = getClusterValues(k)
    features = [feature[0] for feature in labeled_features]
    ac = AgglomerativeClustering(n_clusters=k, affinity='euclidean', linkage='complete')
    y_ac = ac.fit_predict(features)
    pos = 0
    print(y_ac)
    print(len(y_ac))
    for value in y_ac:
        clusters[value].append(labeled_features[pos])
        pos += 1
    return clusters


def getClusterValues(k):
    sklearn_clusters = []
    for i in range(k):
        sklearn_clusters.append([])
    return sklearn_clusters

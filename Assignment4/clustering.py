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


def k_means_clustering(features, k):
    clusters = createClusterGroups(k)
    centroids = getCentroids(features, k)
    distances = config.DEFAULT_DISTANCE_ALGORITHM(features, centroids)
    distances_to_clusters = getDistancesToClusters(distances)
    closest_cluster_idx = assignToCluster(distances_to_clusters)

    for i in range(len(closest_cluster_idx)):
        cluster_num = closest_cluster_idx[i]
        clusters[cluster_num].append(features[i])

    if config.VERBOSE:
        print("Features:\n", features, "\n")
        print("Centroids:\n", centroids, "\n")
        print("Distances:\n", distances, "\n")
        print("Distances to clusters:\n", distances_to_clusters, "\n")
        print("Closest cluster indexes:\n", closest_cluster_idx, "\n")
        for i in range(len(clusters)):
            print("Cluster", i + 1, "\n", clusters[i], "\n")
            print("With", len(clusters[i]), "elements\n")


def createClusterGroups(k):
    clusters = []
    for i in range(k):
        clusters.append([])
    return clusters


def calculateDistance(default_distance_algorithm, *training_data_set, **test_data_set):
    return default_distance_algorithm(*training_data_set, **test_data_set)


def manhattanDistance(training_data_set, test_data_set):
    manhattan_distances = []
    for i in range(len(test_data_set)):
        A = test_data_set[i][0]
        distances = []
        for j in range(len(training_data_set)):
            B = training_data_set[j][0]
            dist = 0
            for k in range(len(A) - 1):
                dist += abs(A[k] - B[k])
            distances.append(dist)
        manhattan_distances.append(distances)
    return manhattan_distances


def getCentroids(features, k):
    centroids = []
    for i in range(k):
        random_centroid_idx = random.randint(0, len(features) - 1)
        while features[random_centroid_idx] in centroids:
            random_centroid_idx = random.randint(0, len(features) - 1)
        centroids.append(features[random_centroid_idx])
    return centroids


def getDistancesToClusters(distances):
    distances_to_clusters = []
    for j in range(len(distances[0])):
        dists = []
        for i in range(len(distances)):
            dists.append(distances[i][j])
        distances_to_clusters.append(dists)
    return distances_to_clusters


def assignToCluster(distances_to_clusters):
    closest_cluster_idx = []
    for distance in distances_to_clusters:
        idx = distance.index(min(distance))
        closest_cluster_idx.append(idx)
    return closest_cluster_idx

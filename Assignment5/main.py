# author        : Esteban
# course        : CS-691 Data Mining
# name          : main.py
# date          : 20191109
# usage         : python3 main.py
# python_version: 3.7
# notes         : Assignment5
# ==============================================================================
import time

import clustering
import config
import sound_tools
import utils


def main():
    execution_times = []
    file_names = utils.loadFileNames()
    mfccs = loadMFCCSValues(file_names)
    labeled_data = utils.createLabeledData(file_names, mfccs)

    # Run sklearn DBSCAN algorithm, write to file and get the k value from it
    start_time = time.time()
    dbscan_clusters = clustering.sklearnDBSCAN(labeled_data)
    execution_times.append(time.time() - start_time)
    utils.writeToFile(config.DBSCAN_KMEANS_OUTPUT, dbscan_clusters)

    results = baselinePerformance(labeled_data, config.NUMBER_OF_CLUSTERS)
    execution_times += results[1]

    print("execution_times:\n", execution_times)

    custom_clusters = results[0][0]
    sklearn_clusters = results[0][1]
    agglomerative_clusters = results[0][2]
    custom_clusters = sorted(custom_clusters)
    sklearn_clusters = sorted(sklearn_clusters)
    agglomerative_clusters = sorted(agglomerative_clusters)

    if config.VERBOSE:
        print("custom_clusters:\n", custom_clusters)
        print("sklearn_clusters:\n", sklearn_clusters)
        print("agglomerative_clusters:\n", agglomerative_clusters)

    goodPerformance(file_names, custom_clusters, sklearn_clusters, agglomerative_clusters)


def loadMFCCSValues(file_names):
    if not utils.fileExists(config.OUTPUT_CSV_FILE):  # if there is no CSV file containing all info
        mfccs = sound_tools.getMFCCS(file_names)
        writing_info = utils.getCSVWriteInfo(file_names, mfccs)
        utils.writeToCSV(writing_info)
        return utils.fixArray(mfccs)
    else:  # if it exists, then just load the values from it instead of computing them again
        return utils.loadCSVInfo()


def baselinePerformance(labeled_data, k):
    execution_times = []

    # CUSTOM K-MEANS CLUSTERING ALGORITHM
    start_time = time.time()
    custom_clusters = clustering.k_means_clustering(labeled_data, k)
    execution_times.append(time.time() - start_time)

    # SKLEARN K-MEANS CLUSTERING ALGORITHM
    start_time = time.time()
    sklearn_clusters = clustering.sklearnKMeansClustering(labeled_data, k)
    execution_times.append(time.time() - start_time)

    # SKLEARN AGGLOMERATIVE CLUSTERING ALGORITHM
    start_time = time.time()
    agglomerative_clusters = clustering.sklearnAgglomerativeClustering(labeled_data, k)
    execution_times.append(time.time() - start_time)

    utils.writeToFile(config.CUSTOM_KMEANS_OUTPUT, custom_clusters)
    utils.writeToFile(config.SKLEARN_KMEANS_OUTPUT, sklearn_clusters)
    utils.writeToFile(config.AGGLOMERATIVE_KMEANS_OUTPUT, agglomerative_clusters)

    if config.DEBUG:
        print("custom_clusters:\n", custom_clusters)
        print("sklearn_clusters:\n", sklearn_clusters)
        print("agglomerative_clusters:\n", agglomerative_clusters)

    # [time1, time2, time3] # time1 = CUSTOM K-MEANS, time2 = SKLEARN K-MEANS, time3 = # SKLEARN AGGLOMERATIVE
    return [custom_clusters, sklearn_clusters, agglomerative_clusters], execution_times


def goodPerformance(file_names, custom_clusters, sklearn_clusters, agglomerative_clusters):
    length = [file_name for a in custom_clusters for file_name in a]
    custom_clusters = utils.getIndexes(file_names, custom_clusters)
    sklearn_clusters = utils.getIndexes(file_names, sklearn_clusters)
    agglomerative_clusters = utils.getIndexes(file_names, agglomerative_clusters)

    matrix = clustering.getMatrix(len(length), custom_clusters, sklearn_clusters, agglomerative_clusters)

    utils.writeMatrixToCSV(length, matrix)


if __name__ == "__main__":
    main()

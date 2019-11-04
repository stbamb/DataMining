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

    execution_times += baselinePerformance(labeled_data, config.NUMBER_OF_CLUSTERS)

    print(execution_times)


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
    return execution_times


if __name__ == "__main__":
    main()

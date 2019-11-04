# author        : Esteban
# course        : CS-691 Data Mining
# name          : main.py
# date          : 20191109
# usage         : python3 main.py
# python_version: 3.7
# notes         : Assignment5
# ==============================================================================
import clustering
import config
import sound_tools
import utils


def main():
    file_names = utils.loadFileNames()
    mfccs = loadMFCCSValues(file_names)
    labeled_data = utils.createLabeledData(file_names, mfccs)

    # Run sklearn DBSCAN algorithm, write to file and get the k value from it
    dbscan_clusters = clustering.sklearnDBSCAN(labeled_data)
    utils.writeToFile(config.DBSCAN_KMEANS_OUTPUT, dbscan_clusters)

    baselinePerformance(labeled_data, len(dbscan_clusters))  # config.NUMBER_OF_CLUSTERS)


def loadMFCCSValues(file_names):
    if not utils.fileExists(config.OUTPUT_CSV_FILE):  # if there is no CSV file containing all info
        mfccs = sound_tools.getMFCCS(file_names)
        writing_info = utils.getCSVWriteInfo(file_names, mfccs)
        utils.writeToCSV(writing_info)
        return utils.fixArray(mfccs)
    else:  # if it exists, then just load the values from it instead of computing them again
        return utils.loadCSVInfo()


def baselinePerformance(labeled_data, k):
    custom_clusters = clustering.k_means_clustering(labeled_data, k)
    sklearn_clusters = clustering.sklearnKMeansClustering(labeled_data, k)
    agglomerative_clusters = clustering.sklearnAgglomerativeClustering(labeled_data, k)

    utils.writeToFile(config.CUSTOM_KMEANS_OUTPUT, custom_clusters)
    utils.writeToFile(config.SKLEARN_KMEANS_OUTPUT, sklearn_clusters)
    utils.writeToFile(config.AGGLOMERATIVE_KMEANS_OUTPUT, agglomerative_clusters)

    if config.DEBUG:
        print("custom_clusters:\n", custom_clusters)
        print("sklearn_clusters:\n", sklearn_clusters)
        print("agglomerative_clusters:\n", agglomerative_clusters)


if __name__ == "__main__":
    main()

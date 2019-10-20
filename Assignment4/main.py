# author        : Esteban
# course        : CS-691 Data Mining
# name          : main.py
# date          : 20191025
# usage         : python3 main.py
# python_version: 3.7
# notes         : Assignment4
# description   :
# ==============================================================================
import clustering
import config
import utils


def main():
    features = utils.gatherFeatures()
    clusters = clustering.k_means_clustering(features, config.NUMBER_OF_CLUSTERS)


if __name__ == "__main__":
    main()

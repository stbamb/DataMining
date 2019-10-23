# author        : Esteban
# course        : CS-691 Data Mining
# name          : main.py
# date          : 20191025
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
    clusters = clustering.k_means_clustering(labeled_features, config.NUMBER_OF_CLUSTERS)

    features = [feature[0] for feature in labeled_features]

    km = KMeans(n_clusters=2, init='random', n_init=1, max_iter=config.MAX_NUMBER_OF_ITERATIONS,
               random_state=0, algorithm='auto')

    y_km = km.fit_predict(features)
    print(y_km)

    print(km.cluster_centers_)

    print(km.inertia_)


if __name__ == "__main__":
    main()

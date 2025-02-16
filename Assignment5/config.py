# author        : Esteban
# course        : CS-691 Data Mining
# name          : config.py
# date          : 20191109
# python_version: 3.7
# notes         : Assignment5
# description   : Needed for main.py to run
# ==============================================================================

NUMBER_OF_FEATURES_TO_EXTRACT = 40
SRC_FOLDER = "data/"
OUTPUT_FOLDER = "output/"
OUTPUT_CSV_FILE = "sound_features.csv"
CUSTOM_KMEANS_OUTPUT = OUTPUT_FOLDER + "custom_kmeans_output.txt"
DBSCAN_KMEANS_OUTPUT_DEFAULT_PARAMETERS = OUTPUT_FOLDER + "dbscan_kmeans_output_default_parameters.txt"
DBSCAN_KMEANS_OUTPUT_EPS = OUTPUT_FOLDER + "dbscan_kmeans_output_eps.txt"
SKLEARN_KMEANS_OUTPUT = OUTPUT_FOLDER + "sklearn_kmeans_output.txt"
AGGLOMERATIVE_KMEANS_OUTPUT = OUTPUT_FOLDER + "agglomerative_output.txt"
MATRIX_OUTPUT = OUTPUT_FOLDER + "matrix.csv"
MAX_NUMBER_OF_ITERATIONS = 5
DBSCAN_EPS = 25
VERBOSE = True
DEBUG = False

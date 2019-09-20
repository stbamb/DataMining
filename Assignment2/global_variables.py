# author        : Esteban
# course        : CS-691 Data Mining
# name          : global_variables.py
# date          : 20190920
# python_version: 3.7
# notes         : Assignment2
# description   : Needed for esteban_murillo_assignment2.py to run
# ==============================================================================
import inhouse_knn

default_distance_algorithm = inhouse_knn.manhattanDistance
image_file_extension = ".jpg"
folder1_image_type = "Insect"
folder2_image_type = "Ocean"
image_folder1 = "Insects/"
image_folder2 = "Ocean/"
max_num_of_k_tries = 10
x_fold = 5
debug = False

# Output messages

extra_algorithms_report_message = "Best value for auto-settings with accuracy mean of {:.4%}, " \
                                  "standard deviation of {:.4} and execution time of {:.4} seconds"

# author        : Esteban
# course        : CS-691 Data Mining
# name          : config.py
# date          : 20191025
# python_version: 3.7
# notes         : Assignment4
# description   : Needed for main.py to run
# ==============================================================================
import clustering

DEFAULT_DISTANCE_ALGORITHM = clustering.manhattanDistance
NUMBER_OF_CLUSTERS = 2
SOURCE_CODE_FOLDER = "SRC/"
JAVA_IMPORT_STATEMENT = "import"
JAVA_SINGLE_LINE_COMMENT = "//"
JAVA_COMMENT_BLOCK_START = "^/*"
JAVA_COMMENT_BLOCK_END = "*/$"
JAVA_COMMENT_BLOCK_REGEX = "/\*(.|[\r\n])*?\*/"
DEBUG = False
VERBOSE = True
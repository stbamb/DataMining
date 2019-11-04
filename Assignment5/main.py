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
    file_names = file_names[:75]

    if not utils.fileExists(config.OUTPUT_CSV_FILE):
        mfccs = sound_tools.getMFCCS(file_names)
        writing_info = utils.getCSVWriteInfo(file_names, mfccs)
        utils.writeToCSV(writing_info)
        mfcss = utils.fixArray(mfccs)
    else:
        mfccs = utils.loadCSVInfo()

    labeled_data = utils.createLabeledData(file_names, mfccs)

    clusters = clustering.k_means_clustering(labeled_data, config.NUMBER_OF_CLUSTERS)

    if config.VERBOSE:
        print("clusters\n", clusters)
        print("len(clusters)", len(clusters))


if __name__ == "__main__":
    main()

# author        : Esteban
# course        : CS-691 Data Mining
# name          : main.py
# date          : 20191109
# usage         : python3 main.py
# python_version: 3.7
# notes         : Assignment5
# ==============================================================================
import config
import sound_tools
import utils


def main():
    file_names = utils.loadFileNames()
    file_names = file_names[:5]

    if not utils.fileExists(config.OUTPUT_CSV_FILE):
        mfccs = sound_tools.getMFCCS(file_names)
        writing_info = utils.getCSVWriteInfo(mfccs)
        utils.writeToCSV(writing_info)

    else:
        mfccs = utils.loadCSVInfo()

    print(mfccs)


if __name__ == "__main__":
    main()

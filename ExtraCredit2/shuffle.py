# author        : Esteban M
# course        : CS-691 Data Mining
# name          : shuffle.py
# date          : 20190913
# usage         : python3 shuffle.py
# python_version: 3.7
# notes         : Assignment for extra credit #2
# description   : Reads data from _input_file_name, shuffles rows
#                 and then outputs it to _output_file_name
# ==============================================================================

import random

_input_file_name = "data_all_with_charge.txt"
_output_file_name = "shuffled_data.txt"


def main():
    lines = readData(_input_file_name)

    if not lines:
        print("Could not read file")
        return

    # code just to preserve the first row, which contains the names of the columns
    first_line = lines[0]
    del lines[0]
    shuffled_lines = shuffle(lines)

    # insert the header row back
    shuffled_lines.insert(0, first_line)
    writeData(_output_file_name, shuffled_lines)


def readData(fileName):
    try:
        with open(fileName) as f:
            lines = f.readlines()
            f.close()
    except:
        return []
    return lines


def writeData(fileName, data):
    try:
        f = open(fileName, "w")
        for i in range(len(data)):
            f.write(data[i])
        f.close()
    except:
        return "Could not write to file"


def shuffle(arr):
    for i in range(len(arr) - 1, -1, -1):
        random_index = random.randint(0, len(arr) - 1)
        temp = arr[random_index]
        arr[random_index] = arr[i]
        arr[i] = temp
    return arr


if __name__ == "__main__":
    main()

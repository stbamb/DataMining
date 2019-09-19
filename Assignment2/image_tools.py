# author        : Esteban
# course        : CS-691 Data Mining
# name          : image_tools.py
# date          : 20190920
# python_version: 3.7
# notes         : Assignment2
# description   : Needed for esteban_murillo_assignment2.py to run
# ==============================================================================

import os

import global_variables as global_vars

from PIL import Image


def loadImages(folder1, folder2):
    full_files_names = []
    directory = os.fsencode(folder1)

    for file in os.listdir(directory):
        file_name = os.fsdecode(file)
        if file_name.endswith(global_vars.image_file_extension):
            full_files_names.append(folder1 + file_name)

    directory = os.fsencode(folder2)

    for file in os.listdir(directory):
        file_name = os.fsdecode(file)
        if file_name.endswith(global_vars.image_file_extension):
            full_files_names.append(folder2 + file_name)

    return full_files_names


def getRGBAttributesForAllFiles(images_names):
    rgb_values = []
    for i in range(len(images_names)):
        rgb_values.append(getRGBAttributes(images_names[i]))
    return rgb_values


def getRGBAttributes(file_name):
    try:
        img = Image.open(file_name)
        img_data = img.load()
    except FileNotFoundError:
        return "Could not find image\n"

    # [R, G, B]
    rgb = [0, 0, 0]
    for x in range(img.size[0]):
        for y in range(img.size[1]):
            r, g, b = img_data[x, y]
            rgb[0] += r
            rgb[1] += g
            rgb[2] += b

    return tuple(rgb)

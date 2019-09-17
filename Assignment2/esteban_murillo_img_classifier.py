# author        : Esteban
# course        : CS-691 Data Mining
# name          : esteban_murillo_img_classifier.py
# date          : 20190920
# usage         : python3 esteban_murillo_img_classifier.py
# python_version: 3.7
# notes         : Assignment2
# description   :
# ==============================================================================

from PIL import Image


_insects_folder = "Insects/"
_ocean_folder = "Ocean/"
_psychonaut = "psychonaut.jpg"


def main():
    rgb = getAvgRGB(_psychonaut)
    print(rgb)


def getAvgRGB(fileName):
    try:
        img = Image.open(fileName)
        imgData = img.load()
    except FileNotFoundError:
        return "Could not find image\n"

    # [R, G, B, TotalPixels]
    rgb = [0, 0, 0, 0]
    for x in range(img.size[0]):
        for y in range(img.size[1]):
            tempr, tempg, tempb = imgData[x, y]
            rgb[0] += tempr
            rgb[1] += tempg
            rgb[2] += tempb
            rgb[3] += 1

    return rgb


if __name__ == "__main__":
    main()

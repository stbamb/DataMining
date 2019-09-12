import random


def shuffle(arr):
    for i in range(len(arr) - 1, -1, -1):
        random_index = random.randint(0, len(arr) - 1)
        temp = arr[random_index]
        arr[random_index] = arr[i]
        arr[i] = temp
    print(arr)

shuffle([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
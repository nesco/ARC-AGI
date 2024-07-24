## Imports
import json
import os
import math

# other imports
import cv2
import numpy as np
from skimage.measure import label, regionprops
import matplotlib.pyplot as plt

## Constants
# Define ANSI escape codes for the closest standard terminal colors
colors = {
    0: "\033[40m",   # black
    1: "\033[44m",   # blue
    2: "\033[101m",  # red
    3: "\033[42m",   # green
    4: "\033[103m",  # yellow
    5: "\033[47m",   # white (for gray, best we can do)
    6: "\033[45m",   # magenta (for fuschia)
    7: "\033[43m",   # dark yellow (for orange, best we can do)
    8: "\033[46m",   # cyan (for teal)
    9: "\033[41m"    # dark red (for brown)
}

## Functions
# Helpers

def read_path(path):
    with open(os.path.join('../data/', path), 'r') as file:
        data = json.load(file)
    return data



def print_colored_grid(grid):
    height = len(grid)
    width = len(grid[0])

    print(f"{colors[5]}   "*(width+2), "\033[0m")

    for row in range(height):

        print(f"{colors[5]}   ", end="")
        for col in range(width):
            # Print with the selected color
            print(f"{colors[grid[row][col]]}   ", end="")
        # Reset color and move to next line
        print(f"{colors[5]}   ", "\033[0m")

    print(f"{colors[5]}   "*(width+2), "\033[0m")


# structure
# - train
#   - []
#       - input [[]]
#       - output [[]]
# - test
#    - [] (len 1)
#      - input [[]]
#      - output [[]]

def filter_by_color(grid, color):
    height = len(grid)
    width = len(grid[0])

    grid_new = [ [0 for j in range(width)] for i in range(height)]

    for i in range(height):
        for j in range(width):
            if grid[i][j] == color:
                grid_new[i][j] = 1

    return grid_new



def split_by_color(grid):
    """Create for each color a binary map of the grid"""
    grids = {}
    for color in range(10):
        grids[color] = filter_by_color(grid, color)
    return grids


# Operator over canals:
def cardinal(canal):
    height = len(canal)
    width = len(canal[0])

    card = sum([1 for i in range(height) for j in range(width) if canal[i][j] == 1])
    return card

# Set operators
def intersection(canal_a, canal_b):
    """Intersection of two similar sized canals"""
    height = len(canal_a)
    width = len(canal_a[0])

    grid_new = [[0 for j in range(width)] for i in range(height)]

    for i in range(height):
        for j in range(width):
            if canal_a[i][j] == 1 and canal_b[i][j] == 1:
                grid_new[i][j] = 1
    return grid_new

def union(canal_a, canal_b):
    """Union of two similar sized canals"""
    height = len(canal_a)
    width = len(canal_a[0])

    grid_new = [[0 for j in range(width)] for i in range(height)]

    for i in range(height):
        for j in range(width):
            if canal_a[i][j] == 1 or canal_b[i][j] == 1:
                grid_new[i][j] = 1
    return grid_new



def complement(canal):
    """Complement of a canal"""
    height = len(canal)
    width = len(canal[0])

    grid_new = [[1 for j in range(width)] for i in range(height)]

    for i in range(height):
        for j in range(width):
            if canal[i][j] == 1:
                grid_new[i][j] = 0
    return grid_new


# Signal-theory operator

def cross_correlation(canal_a, canal_b):
    # Determine which grid is larger and which is the kernel
    if len(canal_a) * len(canal_a[0]) >= len(canal_b) * len(canal_b[0]):
        large = canal_a
        kernel = canal_b
    else:
        large = canal_b
        kernel = canal_a

    large_rows, large_cols = len(large), len(large[0])
    kernel_rows, kernel_cols = len(kernel), len(kernel[0])

    # Dimensions of the correlation matrix
    corr_rows = large_rows - kernel_rows + 1
    corr_cols = large_cols - kernel_cols + 1

    corr_matrix = [[0. for _ in range(corr_rows)] for _ in range(corr_cols)]

    for i in range(corr_rows):
        for j in range(corr_cols):
            sum_prod = 0
            for k in range(kernel_rows):
                for l in range(kernel_cols):
                    # Corresponding coordinates in the large grid
                    y = i + k
                    x = j + l
                    sum_prod += large[x][y] * kernel[k][l]

            sum_kernel = sum([kernel[i][j] for i in range(kernel_rows) for j in range(kernel_cols)])
            #sum_large = sum([large[i][j] for i in range(large_rows) for j in range(large_cols)])
            if  sum_kernel == 0:
                corr_matrix[i][j] = 0
            else:
                corr_matrix[i][j] = sum_prod / sum_kernel

    return corr_matrix


# Set distances and similarities
def jaccard1(canal1, canal2):
    """Jaccard index for two canals of same grid size"""
    rows, cols = len(canal1), len(canal1[0])
    total = rows*cols
    count11 = 0.
    count00 = 0.

    for i in range(rows):
        for j in range(cols):
            if canal1[i][j] == 1 and canal2[i][j] == 1:
                count11 += 1.
            if canal1[i][j] == 0 and canal2[i][j] == 0:
                count00 += 1.

    if count00 == total:
        return 0., -1, -1

    index = count11 / (total - count00)
    return index, -1, -1


def inter_over_union(canal1, canal2):
    canal_intersection = intersection(canal1, canal2)
    canal_union = union(canal1, canal2)

    cardinal_intersection = cardinal(canal_intersection)
    cardinal_union = cardinal(canal_union)

    if cardinal_union == 0:
        return 0., -1, -1
    else:
        return cardinal_intersection / cardinal_union, -1, -1

def correlation_peak(canal1, canal2):
    correlation_matrix = cross_correlation(canal1, canal2)
    max = -1
    max_i, max_j = -1, -1
    for i in range(len(correlation_matrix)):
        for j in range(len(correlation_matrix[0])):
            if correlation_matrix[i][j] > max:
                max = correlation_matrix[i][j]
                max_i, max_j = i,j

    return max, max_i, max_j

def canal_similarity_matrix(similarity, grid1, grid2):
    canals1 = split_by_color(grid1)
    canals2 = split_by_color(grid2)

    similarity_matrix = [[0. for _ in range(10)] for _ in range(10)]

    for i in range(10):
       for j in range(10):
            similarity_matrix[i][j], _, _ = similarity(canals1[i], canals2[j])

    return similarity_matrix
# Signal distance and similarities
## Operators
# Position absolute:
    # Intersection
    # Intersection with pooling
# Position relative
    # Convolution
    # Convolution with pooling
def load_task(task, index):
    data = read_path('training/' + task)
    grid_prob = data['train'][index]['input']
    grid_sol = data['train'][index]['output']
    grids_prob = split_by_color(grid_prob)
    grids_sol = split_by_color(grid_sol)

    return data, grid_prob, grid_sol, grids_prob, grids_sol

# data, grid_prob, grid_sol, grids_prob, grids_sol = load_task(task)

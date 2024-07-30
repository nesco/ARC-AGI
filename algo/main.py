## Imports
import json
import os
import math
from operators import *

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

## Type

class GeometricTree():
    """
    Objects abstracted from a grid can be organized in a tree of concepts under inclusion:
        root: the entire grid
        children: connex components not included in other connex components
        children nth: connex components included in children n-1th
    """
    def __init__(self):
        pass

def constructTree(grid):
    #tree =
    pass
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

def print_binary_grid(grid):
    for row in grid:
        print(' '.join('1' if cell else '0' for cell in row))

def printf_binary_grid(grid):
    bold = '\033[1m'
    reset = '\033[0m'
    for row in grid:
        print(' '.join(f'{bold}1{reset}' if cell else '0' for cell in row))

def colors_extract(grid):
    height, width = len(grid), len(grid[0])
    colors_unique = set()
    for row in range(height):
        for col in range(width):
            colors_unique.add(grid[row][col])

    return colors_unique

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
    """Create for each color a mask, i.e a binary map of the grid"""
    grids = {}
    for color in range(10):
        grids[color] = filter_by_color(grid, color)
    return grids

def extract_masks_bicolors(grid):
    """
    Returns a dict of all masks comprised of union of two colours
    """
    colors_unique = list(colors_extract(grid))
    masks_colors = split_by_color(grid)
    masks_bicolors = {}
    for i in range(len(colors_unique)):
        color_i = colors_unique[i]
        for j in range(i):
            color_j = colors_unique[j]
            masks_bicolors[(color_i, color_j)] = union(masks_colors[color_i], masks_colors[color_j])

        masks_bicolors[(color_i, color_i)] = masks_colors[color_i]
    return masks_bicolors

# Operator over masks:

# Morphological operators

# Signal-theory operator

def cross_correlation(mask1, mask2):
    # Determine which grid is larger and which is the kernel
    if len(mask1) * len(mask1[0]) >= len(mask2) * len(mask2[0]):
        large = mask1
        kernel = mask2
    else:
        large = mask2
        kernel = mask1

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
def jaccard1(mask1, mask2):
    """Jaccard index for two canals of same grid size"""
    rows, cols = len(mask1), len(mask1[0])
    total = rows*cols
    count11 = 0.
    count00 = 0.

    for i in range(rows):
        for j in range(cols):
            if mask1[i][j] == 1 and mask2[i][j] == 1:
                count11 += 1.
            if mask1[i][j] == 0 and mask2[i][j] == 0:
                count00 += 1.

    if count00 == total:
        return 0., -1, -1

    index = count11 / (total - count00)
    return index, -1, -1


def inter_over_union(mask1, mask2):
    canal_intersection = intersection(mask1, mask2)
    canal_union = union(mask1, mask2)

    cardinal_intersection = cardinal(canal_intersection)
    cardinal_union = cardinal(canal_union)

    if cardinal_union == 0:
        return 0., -1, -1
    else:
        return cardinal_intersection / cardinal_union, -1, -1

def correlation_peak(mask1, mask2):
    correlation_matrix = cross_correlation(mask1, mask2)
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
# geometric notion: jaccard
# topologic notion: connex component + correlation


# TO-DO list:
    # Spin number for connected mask
    # inclusion trees for connected masks + top image as root with # of dilatatio
    # euler's number for connected masks
    # density number for connected masks (basically (card(mask) - 2) ) / (card(bounding box full) - 2))
    # where diamond is object of smallest cardinal ccupying a given bounding box
    # Basically two pixels indicating the corners

### Type

def list_components(masks_dict, chebyshev = False):
    components_list = []
    for key in masks_dict:
        mask = masks_dict[key]
        masks_components_seen = [component['mask'] for component in components_list]
        component_number, distribution, components = connected_components(mask, chebyshev=chebyshev)
        components_list += [component for component in components.values()
            if component['mask'] not in masks_components_seen]
    return components_list

def map_contains(masks_list):
    size = len(masks_list)
    comparisons = zeros(size, size)
    for row in range(size):
        for col in range(size):
            comparisons[row][col] = set_contains(masks_list[row], masks_list[col])
    return comparisons

def construct_object_tree(grid):
    height, width = len(grid), len(grid[0])
    masks_bicolors = extract_masks_bicolors(grid)
    #components = {'manhattan': None, 'chebyshev': None}
    # objects = {mask, boundaries, children, topology}
    components_manhattan = list_components(masks_bicolors, chebyshev=False)
    rank_map = {}
    # transpose comparison, so the result reflects the "contained" relation instead of "contains"
    # to calculate the 'rak' of each object
    comparisons = map_contains([component['mask'] for component in components_manhattan])
    comparisons_transposed = transpose(comparisons)
    for i in range(len(components_manhattan)):
        # number of objects containing it, the diagonal is included because it serves a proxy
        # that the object is contained by the grid
        rank = sum(comparisons_transposed[i])
        if rank not in rank_map.keys():
            rank_map[rank] = [i]
        else:
            rank_map[rank].append(i)

    grid_object = {'mask': ones(height, width), 'box': ((0, 0), (height, width)), 'topology': None}
    root = {'value': grid_object, 'children': []}
    nodes = [{'value': component, 'children': []} for component in components_manhattan]

    root['children'] = [(indice, nodes[indice]) for indice in rank_map[1]]

    def complete_tree(children, rank):
        if rank not in rank_map:
            return None

        indices = rank_map[rank]
        for indice, node in children:
            # children are nodes of next rank included in it
            children_indice = [(i, nodes[i]) for i in indices if comparisons[indice][i]]
            node['children'] = complete_tree(children_indice, rank+1)

        return children
    root['children'] = complete_tree(root['children'], 2)
    return root


class Node():
    """
    Objects abstracted from a grid can be organized in a tree of concepts under inclusion:
        root: the entire grid
        children: connex components not included in other connex components
        children nth: connex components included in children n-1th

    Object: rectangular boundary + mask
    """
    def __init__(self, value, children = []):
        self.value = value
        self.children = children

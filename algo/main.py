## Imports
import json
import os
import math

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
    """Create for each color a mask, i.e a binary map of the grid"""
    grids = {}
    for color in range(10):
        grids[color] = filter_by_color(grid, color)
    return grids


# Operator over masks:
def cardinal(mask):
    height = len(mask)
    width = len(mask[0])

    card = sum([1 for i in range(height) for j in range(width) if mask[i][j] == 1])
    return card


# Set operators
def intersection(mask1, mask2):
    """Intersection of two similar sized masks"""
    height = len(mask1)
    width = len(mask1[0])

    mask_new = [[0 for j in range(width)] for i in range(height)]

    for i in range(height):
        for j in range(width):
            if mask1[i][j] == 1 and mask2[i][j] == 1:
                mask_new[i][j] = 1
    return mask_new

def union(mask1, mask2):
    """Union of two similar sized canals"""
    height = len(mask1)
    width = len(mask1[0])

    mask_new = [[0 for j in range(width)] for i in range(height)]

    for i in range(height):
        for j in range(width):
            if mask1[i][j] == 1 or mask2[i][j] == 1:
                mask_new[i][j] = 1
    return mask_new



def complement(mask):
    """Complement of a canal"""
    height = len(mask)
    width = len(mask[0])

    mask_new = [[1 for j in range(width)] for i in range(height)]

    for i in range(height):
        for j in range(width):
            if mask[i][j] == 1:
                mask_new[i][j] = 0
    return mask_new



# Set condition
def contains(mask1, mask2):
    """Is the mask2 contained in mask1?"""
    height = len(mask1)
    width = len(mask1[0])

    for row in range(height):
        for col in range(width):
            if mask2[row][col] == 1 and mask1[row][col] == 0:
                return False
    return True


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

def connex_components(canal, chebyshev=False):
    """Extract connex components.
        Chebyshev: if True then 8-connexity is used instead of 4-connexity
    """
    component_number = 0
    height, width = len(canal), len(canal[0])
    distribution = [[0 for _ in range(width)] for _ in range(height)]
    to_see = [(i,j) for j in range(width) for i in range(height)]
    seen = set()
    components = {}
    #bounding_boxes = {}

    for i,j in to_see:
        # Check if there is something to do
        if (i,j) in seen or canal[i][j] == 0:
            continue

        # New component found, increase the count and initiate queue
        component_number += 1
        components[component_number] = {'mask': [[0 for _ in range(width)] for _ in range(height)]}
        queue = [(i, j)]
        min_row, max_row = i, i
        min_col, max_col = j, j

        while queue:
            k, l = queue.pop(0)  # Correctly manage the queue
            if (k, l) in seen:
                continue

            distribution[k][l] = component_number
            components[component_number]['mask'][k][l] = 1
            seen.add((k, l))

            # Update the bounding box
            min_row, max_row = min(min_row, k), max(max_row, k)
            min_col, max_col = min(min_col, l), max(max_col, l)

            # Add all 4-connex neighbors
            for dk, dl in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                nk, nl = k + dk, l + dl
                if 0 <= nk < height and 0 <= nl < width and canal[nk][nl] == 1 and (nk, nl) not in seen:
                    queue.append((nk, nl))

            if chebyshev:
                # Add diagonal neighbors
                for dk, dl in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                    nk, nl = k + dk, l + dl
                    if 0 <= nk < height and 0 <= nl < width and canal[nk][nl] == 1 and (nk, nl) not in seen:
                        queue.append((nk, nl))

        # Store the bounding box for the current component
        components[component_number]['box'] = ((min_row, min_col), (max_row, max_col))
        #bounding_boxes[component_number] = ((min_row, min_col), (max_row, max_col))

    return component_number, distribution, components #bounding_boxes
# component_number, distribution, components = connex_components(grids_sol[4])
#def contains(mask1, mask2):
    """Is the mask2 contained in mask1?"""
    height = len(mask1)
    width = len(mask1[0])

    for row in range(height):
        for col in range(width):
            if mask2[row][col] == 1 and mask1[row][col] == 0:
                return False
    return True

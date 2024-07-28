"""This module contains all pixel-level operators on binary masks.
Binary masks are square grids, represented by lists of lists, of 0s (Nothing) and 1s (Something).

Operators are classified in:
    - Set operators: operators that only act at pixel level, without any interaction
    - Geometric operators: operators that involves grid information
    - Morphological operators: they involve notions of kernel
    - Topological operators: they involve notions of connectivity.
        Either 4-connectivity (Manhattan) or 8-connectivity (Chebyshev)
    - Signal and probabilistic operators: [TO-DO]
"""

### Constants
# Topological constants
# The Manhattan / 4-connectivity kernel is a cross
kernel_manhattan = [[0, 1, 0], [1, 1, 1], [0, 1, 0]]
# The Chebyshev / 8-connectivity kernel is a square
kernel_chebyshev = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]

### Operators on binary masks
## Helpers
def flatten(mask):
    height, width = len(mask), len(mask[0])
    return [mask[row][col] for row in range(height) for col in range(width)]

def zeros(height, width):
    return [[0 for _ in range(width)] for _ in range(height)]

def ones(height, width):
    return [[1 for _ in range(width)] for _ in range(height)]

def identity(height, width):
    return [[1 if col == row else 0 for col in range(width)] for row in range(height)]

def fill_unary(mask, condition):
    height = len(mask)
    width = len(mask[0])

    mask_new = zeros(height, width)
    for row in range(height):
        for col in range(width):
            mask_new[row][col] = condition(mask[row][col])
    return mask_new

def fill_binary(mask1, mask2, condition):
    height = len(mask1)
    width = len(mask1[0])

    mask_new = zeros(height, width)
    for row in range(height):
        for col in range(width):
            if condition(mask1[row][col], mask2[row][col]):
                mask_new[row][col] = 1
    return mask_new

## Set operators
# Set Mapping
def cardinal(mask):
    return sum(flatten(mask))

def intersection(mask1, mask2):
    """Intersection of two similar sized masks"""
    return fill_binary(mask1, mask2, lambda a, b: (a == 1) and (b == 1))

def union(mask1, mask2):
    """Union of two similar sized canals"""
    return fill_binary(mask1, mask2, lambda a, b: (a == 1) or (b == 1))

def complement(mask):
    return fill_unary(mask, lambda a: 0 if (a == 1) else 1)
# Set conditions
def contains(mask_larger, mask_smaller):
    return intersection(mask_larger, mask_smaller) == mask_smaller
## Geometric operators
def transpose(mask):
    height, width = len(mask), len(mask[0])
    return [[mask[row][col] for row in range(height)] for col in range(width)]

def reverse_rows(mask):
    height, width = len(mask), len(mask[0])
    return [[mask[row][col] for col in range(height)] for row in reversed(range(width))]

def reverse_cols(mask):
    height, width = len(mask), len(mask[0])
    return [[mask[row][col] for col in reversed(range(height))] for row in range(width)]

def rotation(mask):
    return transpose(reverse_cols(mask))

## Morphological operators
def morphological_dilatation(mask_object, mask_kernel):
    """Returns the mask of the cropped morphological dilatation (~Minkowski addition) of an object and a structuring element """
    object_height, object_width = len(mask_object), len(mask_object[0])
    kernel_height, kernel_width = len(mask_kernel), len(mask_kernel[0])

    # The new mask will have the dimension of the object's mask
    # It's the only asymetry between "object" and "kernel"
    # Whereas traditional Minkowski addition treats them equally
    mask_new = zeros(object_height, object_width)

    for row in range(object_height):
        for col in range(object_width):
            if mask_object[row][col] == 1:
                for i in range(kernel_height):
                    for j in range(kernel_width):
                        k = row + i - kernel_height // 2
                        l = col + j - kernel_width // 2

                        if 0 <= k < object_height and 0 <= l < object_width:
                            if mask_kernel[i][j] == 1:
                                mask_new[k][l] = 1
    return mask_new


def morphological_erosion(mask_object, mask_kernel):
    """Returns the mask of the morphological erosion of an object using a structuring element."""
    object_height, object_width = len(mask_object), len(mask_object[0])
    kernel_height, kernel_width = len(mask_kernel), len(mask_kernel[0])

    # The new mask will have the dimension of the object's mask
    mask_new = zeros(object_height, object_width)

    for row in range(object_height):
        for col in range(object_width):
            # Check if the kernel fits
            fits = True
            for i in range(kernel_height):
                for j in range(kernel_width):
                    k = row + i - kernel_height // 2
                    l = col + j - kernel_width // 2

                    if 0 <= k < object_height and 0 <= l < object_width:
                        if mask_kernel[i][j] == 1 and mask_object[k][l] == 0:
                            fits = False
                            break
                    else:
                        fits = False
                        break
                if not fits:
                    break

            if fits:
                mask_new[row][col] = 1
            else:
                mask_new[row][col] = 0

    return mask_new

def morphological_opening(mask_object, mask_kernel):
    return morphological_dilatation(morphological_erosion(mask_object, mask_kernel), mask_kernel)

def morphological_closing(mask_object, mask_kernel):
    return morphological_erosion(morphological_dilatation(mask_object, mask_kernel), mask_kernel)
## Topological operators
def dilatation(mask, chebyshev = False):
    if chebyshev:
        return morphological_dilatation(mask, kernel_chebyshev)
    return morphological_dilatation(mask, kernel_manhattan)

def erosion(mask, chebyshev = False):
    if chebyshev:
        return morphological_erosion(mask, kernel_chebyshev)
    return morphological_erosion(mask, kernel_manhattan)

def opening(mask, chebyshev = False):
    if chebyshev:
        return morphological_opening(mask, kernel_chebyshev)
    return morphological_opening(mask, kernel_manhattan)

def closing(mask, chebyshev = False):
    if chebyshev:
        return morphological_closing(mask, kernel_chebyshev)
    return morphological_closing(mask, kernel_manhattan)

#### Functions of binary masks
## Topological
def connex_components(mask, chebyshev=False):
    """Extract connex components.
        Chebyshev: if True then 8-connexity is used instead of 4-connexity
    """
    component_number = 0
    height, width = len(mask), len(mask[0])
    distribution = zeros(height, width)
    to_see = [(i,j) for j in range(width) for i in range(height)]
    seen = set()
    components = {}

    for i,j in to_see:
        # Check if there is something to do
        if (i,j) in seen or mask[i][j] == 0:
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
                if 0 <= nk < height and 0 <= nl < width and mask[nk][nl] == 1 and (nk, nl) not in seen:
                    queue.append((nk, nl))

            if chebyshev:
                # Add diagonal neighbors
                for dk, dl in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                    nk, nl = k + dk, l + dl
                    if 0 <= nk < height and 0 <= nl < width and mask[nk][nl] == 1 and (nk, nl) not in seen:
                        queue.append((nk, nl))

        # Store the bounding box for the current component
        components[component_number]['box'] = ((min_row, min_col), (max_row, max_col))

    return component_number, distribution, components

## Tests
def _test_flatten():
    assert flatten([[1, 1], [0, 1]]) == [1, 1, 0, 1]

def _test_intersection():
    assert intersection([[1, 1],[0, 1]], [[1, 0], [1, 1]]) == [[1, 0], [0, 1]]
    assert intersection([[1, 0], [0, 1]], [[0, 1],[1, 0]]) == [[0, 0], [0, 0]]

def _test_union():
    assert union([[1, 0], [0, 1]], [[0, 1],[1, 0]]) == [[1, 1], [1, 1]]
    assert union([[1, 0], [0, 0]], [[0, 0], [0, 1]]) == [[1, 0], [0, 1]]

def _test_complement():
    assert complement([[1, 0], [0, 1]]) == [[0, 1], [1, 0]]

def _test_transpose():
    assert transpose([[1, 0], [0, 1]]) == [[1, 0], [0, 1]]
    assert transpose([[0, 1], [0, 0]]) == [[0, 0], [1, 0]]
    assert transpose([[0, 0], [1, 0]]) == [[0, 1], [0, 0]]

def _test_reverse_cols():
    assert reverse_cols([[1, 0], [1, 0]]) == [[0, 1], [0, 1]]
    assert reverse_cols([[0, 1], [0, 1]]) == [[1, 0], [1, 0]]

def _test_reverse_rows():
    assert reverse_rows([[1, 1], [0, 0]]) == [[0, 0], [1, 1]]
    assert reverse_rows([[0, 0], [1, 1]]) == [[1, 1], [0, 0]]

def display_pixels_on_empty_grid(points):

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

    grid = [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ]

    for x, y in points:
        grid[x][y] = 1

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
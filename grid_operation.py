from utils import display_pixels_on_empty_grid

class GridOperation:
    def __init__(self):
        pass

    def _get_bounding_box(self, pixels):
        x_coords, y_coords = zip(*pixels)
        return min(x_coords), max(x_coords), min(y_coords), max(y_coords)

    def rotate_90(self, pixels):
        original = frozenset(pixels)
        min_x, max_x, min_y, max_y = self._get_bounding_box(original)
        width, _ = max_x - min_x, max_y - min_y
        return frozenset((min_x + (p[1] - min_y), min_y + (width - (p[0] - min_x))) for p in original)

    def flip_horizontal(self, pixels):
        original = frozenset(pixels)
        min_x, max_x, _, _ = self._get_bounding_box(original)
        return frozenset((min_x + max_x - p[0], p[1]) for p in original)

    def flip_vertical(self, pixels):
        original = frozenset(pixels)
        _, _, min_y, max_y = self._get_bounding_box(original)
        return frozenset((p[0], min_y + max_y - p[1]) for p in original)


if __name__ == "__main__":
    order_name = ["original", "r90", "r180", "r270", "h", "hr90", "hr180", "hr270", "v"]

    grid_op = GridOperation()

    original = {(1,1), (1,2), (1,3), (2,2), (2,3)}

    r90 = grid_op.rotate_90(original)
    r180 = grid_op.rotate_90(r90)
    r270 = grid_op.rotate_90(r180)

    h = grid_op.flip_horizontal(original)

    hr90 = grid_op.rotate_90(h)
    hr180 = grid_op.rotate_90(hr90)
    hr270 = grid_op.rotate_90(hr180)

    v = grid_op.flip_vertical(original)

    print("original")
    display_pixels_on_empty_grid(original)
    print("r90")
    display_pixels_on_empty_grid(r90)
    print("r180")
    display_pixels_on_empty_grid(r180)
    print("r270")
    display_pixels_on_empty_grid(r270)
    print("h")
    display_pixels_on_empty_grid(h)
    print("hr90")
    display_pixels_on_empty_grid(hr90)
    print("hr180")
    display_pixels_on_empty_grid(hr180)
    print("hr270")
    display_pixels_on_empty_grid(hr270)
    print("v")
    display_pixels_on_empty_grid(v)
    
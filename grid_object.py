import numpy as np
from scipy.spatial import distance

class GridObject:
    def __init__(self, pixels):
        # pixels est une liste de tuples (x, y) qui représentent les coordonnées des pixels
        self.pixels = set(pixels)
        self._update_properties()
    
    def _update_properties(self):
        # Met à jour les propriétés de l'objet
        self._calculate_dimensions()
        self._calculate_center_of_mass()
        self._calculate_perimeter()
        self._get_bounding_box_pixels()
    
    def _calculate_dimensions(self):
        xs, ys = zip(*self.pixels)
        self.width = max(xs) - min(xs) + 1
        self.height = max(ys) - min(ys) + 1
        self.area = len(self.pixels)
    
    def _calculate_center_of_mass(self):
        xs, ys = zip(*self.pixels)
        self.center_of_mass = (np.mean(xs), np.mean(ys))
    
    def _calculate_perimeter(self):
        # Implémentation simplifiée pour des formes simples
        def is_edge(p1, p2):
            return p1 in self.pixels and p2 in self.pixels

        perimeter = 0
        for (x, y) in self.pixels:
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                if not is_edge((x + dx, y + dy), (x, y)):
                    perimeter += 1
        self.perimeter = perimeter

    def _get_bounding_box_pixels(self):
        """
        Calculates the bounding box for a set of pixels and returns the bounding box coordinates.

        :param pixels: Set of pixel coordinates (x, y).
        :return: Set of pixel coordinates forming the bounding box.
        """
        pixels = self.pixels
        if not pixels:
            return set()

        # Calculate the bounding box
        min_x = min(x for x, y in pixels)
        max_x = max(x for x, y in pixels)
        min_y = min(y for x, y in pixels)
        max_y = max(y for x, y in pixels)

        # Create the set of bounding box pixels using set comprehensions
        self.bouding_box_pixels = {(x, y) for x in range(min_x, max_x + 1) for y in range(min_y, max_y + 1)}
    
    def get_position(self):
        return min(self.pixels, key=lambda p: (p[1], p[0]))

    def intersect(self, other):
        # Retourne True si l'objet intersect avec un autre objet
        return not self.pixels.isdisjoint(other.pixels)
    
    def distance_to(self, other):
        # Calcul de la distance entre les centres de gravité des objets
        return distance.euclidean(self.center_of_mass, other.center_of_mass)
    
    def translate(self, dx, dy):
        # Translate the object by (dx, dy)
        translated_pixels = {(x + dx, y + dy) for x, y in self.pixels}
        return GridObject(list(translated_pixels))
    
    def is_translation_of(self, other):
        if self.area != other.area:
            return False
        
        # Generate all possible translations
        for (x1, y1) in self.pixels:
            for (x2, y2) in other.pixels:
                dx, dy = x2 - x1, y2 - y1
                translated_self = self.translate(dx, dy)
                if translated_self.pixels == other.pixels:
                    return True
        return False
    
    def contains_object(self, contained_object):
        # Convertir les ensembles de pixels en sets pour la comparaison
        container_set = self.pixels
        contained_set = contained_object.pixels
        
        # Vérifier si tous les pixels de l'objet à tester sont dans l'objet contenant
        return contained_set.issubset(container_set)
    
    def is_contained_by_object(self, container_object):
        # Convertir les ensembles de pixels en sets pour la comparaison
        container_set = container_object.pixels
        contained_set = self.pixels
        
        # Vérifier si tous les pixels de l'objet à tester sont dans l'objet contenant
        return contained_set.issubset(container_set)
    
    def is_touching(self, touching_object):
        """aims to see if two objects are connected
        - is touching
        similar to
        - is touched by
        - connected with
        """
        pixels1 = self.pixels
        pixels2 = touching_object.pixels
        # Directions possibles pour les voisins (horizontaux, verticaux, diagonaux)
        directions = [(-1, -1), (-1, 0), (-1, 1),
                    (0, -1),          (0, 1),
                    (1, -1), (1, 0), (1, 1)]
        
        # Vérifie si un pixel dans `pixels1` est adjacent à un pixel dans `pixels2`
        for x1, y1 in pixels1:
            for dx, dy in directions:
                if (x1 + dx, y1 + dy) in pixels2:
                    return True
        
        return False
    
    def greater(self, object_to_compare):
        size1 = len(self.pixels)
        size2 = len(object_to_compare.pixels)
        
        if size1 > size2:
            return True
        else:
            return False
        
    def smaller(self, object_to_compare):
        size1 = len(self.pixels)
        size2 = len(object_to_compare.pixels)
        
        if size1 < size2:
            return True
        else:
            return False
        
    def equal(self, object_to_compare):
        size1 = len(self.pixels)
        size2 = len(object_to_compare.pixels)
        
        if size1 == size2:
            return True
        else:
            return False

    
    def __str__(self):
        return (f"Position: {self.get_position()}, "
                f"Dimensions: ({self.width}, {self.height}), "
                f"Area: {self.area}, "
                f"Center of Mass: {self.center_of_mass}, "
                f"Perimeter: {self.perimeter}")


if __name__ == "__main__":
    # Exemple d'utilisation
    container_pixels = GridObject([(0,0), (0,1), (1,0), (1,1), (2,0), (2,1)])
    contained_pixels = GridObject([(0,0), (0,1), (1,0), (1,1)])

    print(container_pixels.contains_object(contained_pixels))  # Devrait afficher True
    print(container_pixels.is_contained_by_object(contained_pixels))  # Devrait afficher True

    # Exemple d'utilisation
    pixels_object1 = GridObject([(0,0), (0,1), (1,0), (1,1)])
    pixels_object2 = GridObject([(1,1), (2,1), (2,2)])
    pixels_object3 = GridObject([(3,3), (4,4)])

    print(pixels_object1.is_touching(pixels_object2))  # Devrait afficher True (ils se touchent)
    print(pixels_object1.is_touching(pixels_object3))  # Devrait afficher False (ils ne se touchent pas)

    # Exemple d'utilisation
    pixels_object1 = GridObject([(0,0), (0,1), (1,0), (1,1)]) # Taille: 4
    pixels_object2 = GridObject([(1,1), (2,1), (2,2)])        # Taille: 3
    pixels_object3 = GridObject([(3,3), (4,4)])               # Taille: 2

    print(pixels_object1.greater(pixels_object1))
    print(pixels_object1.greater(pixels_object2))
    print(pixels_object1.greater(pixels_object3))

    print(pixels_object1.smaller(pixels_object1))
    print(pixels_object1.smaller(pixels_object2))
    print(pixels_object1.smaller(pixels_object3))

    print(pixels_object1.equal(pixels_object1))
    print(pixels_object1.equal(pixels_object2))
    print(pixels_object1.equal(pixels_object3))

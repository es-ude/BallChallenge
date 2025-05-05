import math
import numpy as np

class Vector:
    def __init__(self, x, y):
        self._x = x
        self._y = y

    def __repr__(self):
        return f"Vector({self._x}, {self._y})"

    def __eq__(self, other):
            return self._x == other._x and self._y == other._y

    @property
    def get_x(self):
        return self._x

    @property
    def get_y(self):
        return self._y

    def normalize(self):
       return Vector(self._x / math.sqrt(self._x**2 + self._y**2), self._y / math.sqrt(self._x**2 + self._y**2))

    def length(self):
        return math.sqrt((self._x)**2 + (self._y)**2)

    def dot_product(self, other):
        dot = self._x * other._x + self._y * other._y
        return max(-1.0, min(1.0, dot))

    def sub(self, other):
        return Vector(self._x - other._x, self._y - other._y)

    def sub_x_coordinates(self, other):
        return self._x - other._x

    def sub_y_coordinates(self, other):
        return self._y - other._y

    def swap(self):
        return Vector(self._y, self._x)

    def mult_with_y_component(self, number):
        return Vector(self._x, self._y * number)

    def add(self, other):
        return Vector(self._x + other._x, self._y + other._y)

    def add_with_x_coordinate(self, number):
        return Vector(self._x + number, self._y)

    def add_with_y_coordinate(self, number):
        return Vector(self._x, self._y + number)

    def mult(self, other):
        return Vector(self._x * other._x, self._y * other._y)

    def div(self, number):
        return Vector(self._x / number, self._y / number)

    def component_addition(self):
        return self._x + self._y

    def component_subtraction(self):
        return self._x - self._y

    def potentiate_vector(self):
        return Vector(self._x * self._x, self._y * self._y)

    def is_horizontal(self, other):
        return self._y == other._y

    def is_vertical(self, other):
        return self._x == other._x

    def round_vector(self, decimal_places):
        return Vector(round(self._x, decimal_places), round(self._y, decimal_places))

    def sort_Vectors(self, other):
        if self._x < other._x:
            return self, other
        elif self._x == other._x:
            if self._y <= other._y:
                return self, other
            else:
                return other, self
        else:
            return other, self

class Line:
    def __init__(self, a: Vector, b: Vector):
        self._a, self._b = a.sort_Vectors(b)

    def __repr__(self):
        return f"Line({self._a}, {self._b})"

    def __eq__(self, other):
        return self._a.__eq__(other._a) and self._b.__eq__(other._b)

    def normal_vector(self):
        return self._a.sub(self._b).mult_with_y_component(-1).swap().normalize()

    def distance(self):
        return self.normal_vector().mult(self._a).component_addition()

    def length(self):
        return math.sqrt(self._b.sub(self._a).potentiate_vector().component_addition())

    def is_point_on_line(self, point):
        return abs((self.normal_vector().mult(point)).component_addition() - self.distance()) < 10e-2

    def mid_point(self):
        return self._a.add(self._b).div(2)

    def compute_degree_of_two_normal_vectors_to_each_other(self, other):
        normal_vector_one = self.normal_vector()
        normal_vector_two = other.normal_vector()
        dot_product = normal_vector_one.dot_product(normal_vector_two)
        return math.degrees(math.acos(dot_product))

    def is_parallel(self, other, tolerance=5):
        angle = self.compute_degree_of_two_normal_vectors_to_each_other(other)
        return abs(angle) < tolerance or abs(angle - 180) < tolerance

    def compute_y_coordinate(self, x):
        return (self.distance() - x * self.normal_vector().get_x) / self.normal_vector().get_y

    def get_points_from_line(self, amount_of_points):
        self.step_size = self.length() / (amount_of_points - 1)
        self.step_size_x = self._b.sub_x_coordinates(self._a) / (amount_of_points - 1)
        points_on_line = []
        points_on_line.append(self._a)
        if self._a.is_vertical(self._b):
            for i in range(0, amount_of_points - 1):
                points_on_line.append(points_on_line[i].add_with_y_coordinate(self.step_size))
            return points_on_line
        elif self._a.is_horizontal(self._b):
            for i in range(0, amount_of_points - 1):
                points_on_line.append(points_on_line[i].add_with_x_coordinate(self.step_size))
            return points_on_line
        else:
            for i in range(0, amount_of_points - 1):
                points_on_line.append(Vector(points_on_line[i].get_x + self.step_size_x, self.compute_y_coordinate(points_on_line[i].get_x + self.step_size_x)))
            return points_on_line

    def intersection(self, other):
        return Vector(((self.distance() * other.normal_vector().get_y) - (other.distance() * self.normal_vector().get_y)) / ((self.normal_vector().get_x * other.normal_vector().get_y) - (other.normal_vector().get_x * self.normal_vector().get_y)),
                      ((self.normal_vector().get_x * other.distance()) - (other.normal_vector().get_x * self.distance())) / ((self.normal_vector().get_x * other.normal_vector().get_y) - (other.normal_vector().get_x * self.normal_vector().get_y)))

    def calculate_angle_of_normal_vector(self):
        theta = np.arctan2(self.normal_vector().get_y, self.normal_vector().get_x)
        return np.degrees(theta) % 180

class Grid:
    def __init__(self, side_lines, grid_size):
        self._side_lines = side_lines
        self._grid_size = grid_size

    def create_inner_lines(self):
        inner_lines = []
        for i in self._side_lines:
            inner_lines.append(i)

        for i in range(len(self._side_lines)):
            for j in range(i + 1, len(self._side_lines)):
                if self._side_lines[i].is_parallel(self._side_lines[j], 10):
                    upper_grid_points = self._side_lines[i].get_points_from_line(self._grid_size)
                    under_grid_points = self._side_lines[j].get_points_from_line(self._grid_size)
                    for k in range(1, len(upper_grid_points) - 1):
                        inner_lines.append(Line(upper_grid_points[k].round_vector(2), under_grid_points[k].round_vector(2)))
        return inner_lines

    def calculate_grid_points(self):
        self.all_grid_points = []
        self.grid_points = []
        self.all_lines = []
        for i in self._side_lines:
            self.all_lines.append(i)

        for j in self.create_inner_lines():
            self.all_lines.append(j)

        for k in range(len(self.all_lines)):
            for l in range(k + 1, len(self.all_lines)):
                if self.all_lines[k].is_parallel(self.all_lines[l], 10) == False:
                    self.all_grid_points.append(self.all_lines[k].intersection(self.all_lines[l]).round_vector(2))

        for m in self.all_grid_points:
            if m not in self.grid_points:
                self.grid_points.append(m)
        return self.grid_points

class Square:
    def __init__(self, lines):
        self.lines = lines

    def __repr__(self):
        return f"Quadrat({self.lines[0]}, {self.lines[1]}, {self.lines[2]}, {self.lines[3]})"

    def get_corner_coords(self):
        corner_points = []
        for i in range(1, 4):
            if self.lines[0].is_parallel(self.lines[i]):
                corner_points.append((self.lines[0]._a.get_x, self.lines[0]._a.get_y))
                corner_points.append((self.lines[0]._b.get_x, self.lines[0]._b.get_y))
                corner_points.append((self.lines[i]._a.get_x, self.lines[i]._a.get_y))
                corner_points.append((self.lines[i]._b.get_x, self.lines[i]._b.get_y))
        return corner_points

    def is_point_inside(self, point):
        max_x = max(x for x, _ in self.get_corner_coords())
        min_x = min(x for x, _ in self.get_corner_coords())
        max_y = max(y for _, y in self.get_corner_coords())
        min_y = min(y for _, y in self.get_corner_coords())
        return min_x < point.get_x < max_x and min_y < point.get_y < max_y

    def points_inside_square(self, list_of_points):
        points_in_square = []
        for i in list_of_points:
            if self.is_point_inside(i):
                points_in_square.append(i)
        return points_in_square

class Circle:
    def __init__(self, center, radius):
        self.center = center
        self.radius = radius

    def is_point_in_circle(self, point):
        return point.sub(self.center).potentiate_vector().component_addition() < self.radius ** 2

    def amount_of_points_in_circle(self, points):
        amount_of_points = 0
        for i in points:
            if self.is_point_in_circle(i):
                amount_of_points = amount_of_points + 1
        return amount_of_points - 1



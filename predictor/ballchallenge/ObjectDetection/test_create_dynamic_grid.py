import pytest
import math
from create_dynamic_grid import Vector, Line, Grid, Square, Circle

def test_normalized_vector():
    vector = Vector(5, 8)
    norminated_vector = vector.normalize()
    assert math.sqrt(norminated_vector.potentiate_vector().component_addition()) <= 1

def test_calculate_distance():
    vector_one = Vector(1, 1)
    vector_two = Vector(2, 2)
    assert Line(vector_one, vector_two).distance() <= 0

def test_calculate_line_length():
    vector_one = Vector(0, 0)
    vector_two = Vector(1, 1)
    assert Line(vector_one, vector_two).length() <= math.sqrt(2)

def test_vector_length():
    vector = Vector(1, 1)
    vector_length = vector.length()
    assert vector_length <= math.sqrt(2)

def test_calculate_dot_product():
    vector_one = Vector(1, 1)
    vector_two = Vector(2, 1)
    assert vector_one.dot_product(vector_two) <= 3

def test_calculate_parallel_lines_for_parallel_lines():
    vector_one = Vector(1, 1)
    vector_two = Vector(1, 2)
    vector_three = Vector(2, 1)
    vector_four = Vector(2, 2)
    line_one = Line(vector_one, vector_two)
    line_two = Line(vector_three, vector_four)
    assert Line.compute_degree_of_two_normal_vectors_to_each_other(line_one, line_two) <= 0

def test_calculate_parallel_lines_for_orthogonal_lines():
    vector_one = Vector(1, 2)
    vector_two = Vector(2, 2)
    vector_three = Vector(2, 1)
    vector_four = Vector(2, 2)
    line_one = Line(vector_one, vector_two)
    line_two = Line(vector_three, vector_four)
    assert Line.compute_degree_of_two_normal_vectors_to_each_other(line_one, line_two) <= 90

def test_is_parallel():
    assert Line(Vector(1, 2), Vector(4, 4)).is_parallel(Line(Vector(1, 1), Vector(4, 3))) <= True

def test_is_not_parallel():
    assert Line(Vector(1, 2), Vector(4, 4)).is_parallel(Line(Vector(1, 1), Vector(1, 4))) <= False

def test_is_point_on_line():
    assert Line(Vector(1, 1), Vector(3, 3)).is_point_on_line(Vector(2, 2)) is True

def test_is_point_not_on_line():
    assert Line(Vector(1, 1), Vector(3, 3)).is_point_on_line(Vector(2, 3)) is False

def test_compute_y_coordinate():
    assert Line(Vector(1, 1), Vector(3, 3)).compute_y_coordinate(2) <= 2

def test_get_points_from_line_for_vertical_line():
    expected_output = "[Vector(1, 1), Vector(1, 2.0), Vector(1, 3.0), Vector(1, 4.0)]"
    assert repr(Line(Vector(1, 1), Vector(1, 4)).get_points_from_line(4)) == expected_output

def test_gets_points_from_line_for_horizontal_line():
    expected_output = "[Vector(1, 1), Vector(3.5, 1), Vector(6.0, 1), Vector(8.5, 1), Vector(11.0, 1)]"
    assert repr(Line(Vector(1, 1), Vector(11, 1)).get_points_from_line(5)) == expected_output

def test_get_points_from_line_for_diagonal_line():
    expected_output = "[Vector(1, 1), Vector(2.5, 2.5), Vector(4.0, 4.0)]"
    assert repr(Line(Vector(1, 1), Vector(4, 4)).get_points_from_line(3)) == expected_output

def test_get_points_from_line_for_diagonal_line_sec():
    expected_output = "[Vector(1, 1), Vector(2.0, 2.0), Vector(3.0, 3.0), Vector(4.0, 4.0)]"
    assert repr(Line(Vector(1, 1), Vector(4, 4)).get_points_from_line(4)) == expected_output

def test_create_inner_lines():
    expected_output = "[Line(Vector(1, 2), Vector(1, 4)), Line(Vector(1, 4), Vector(3, 3)), Line(Vector(3, 1), Vector(3, 3)), Line(Vector(1, 2), Vector(3, 1)), Line(Vector(1, 3.0), Vector(3, 2.0)), Line(Vector(2.0, 1.5), Vector(2.0, 3.5))]"
    assert repr(Grid([Line(Vector(1, 2), Vector(1, 4)), Line(Vector(1, 4), Vector(3, 3)),
                      Line(Vector(3, 1), Vector(3, 3)), Line(Vector(3, 1), Vector(1, 2))], 3).create_inner_lines()) == expected_output

def test_intersection():
    expected_output = "Vector(3.0, 3.0)"
    assert repr(Line(Vector(3, 1), Vector(3, 5)).intersection(Line(Vector(1, 3), Vector(5, 3)))) == expected_output

def test_intersection_and_compute_y_coordinates():
    expected_output = f"Vector(2.5, {Line(Vector(1, 1), Vector(4, 4)).compute_y_coordinate(2.5)})"
    assert repr(Line(Vector(1, 1), Vector(4, 4)).intersection(Line(Vector(1, 4), Vector(4, 1)))) == expected_output

def test_calculate_grid_points():
    expected_output = "[Vector(2.0, 4.0), Vector(1.0, 1.0), Vector(1.5, 2.5), Vector(4.0, 4.0), Vector(3.0, 4.0), Vector(3.0, 1.0), Vector(3.5, 2.5), Vector(2.0, 1.0), Vector(2.5, 2.5)]"
    assert repr(Grid([Line(Vector(1, 1), Vector(2, 4)), Line(Vector(2, 4), Vector(4, 4)),
                      Line(Vector(4, 4), Vector(3, 1)), Line(Vector(3, 1), Vector(1, 1))], 3).calculate_grid_points()) == expected_output

def test_calculate_grid_points_for_2_times_2_grid():
    expected_output = "[Vector(2.0, 4.0), Vector(1.0, 1.0), Vector(1.33, 2.0), Vector(1.67, 3.0), Vector(4.0, 4.0), Vector(2.67, 4.0), Vector(3.33, 4.0), Vector(3.0, 1.0), Vector(3.33, 2.0), Vector(3.67, 3.0), Vector(1.67, 1.0), Vector(2.33, 1.0), Vector(2.0, 2.0), Vector(2.66, 2.0), Vector(2.34, 3.0), Vector(3.0, 3.0)]"
    assert repr(Grid([Line(Vector(1, 1), Vector(2, 4)), Line(Vector(2, 4), Vector(4, 4)),
                      Line(Vector(4, 4), Vector(3, 1)), Line(Vector(3, 1), Vector(1, 1))], 4).calculate_grid_points()) == expected_output

def test_calculate_angle_of_normal_vector():
    assert Line(Vector(1, 1), Vector(4, 1)).calculate_angle_of_normal_vector() == 90

def test_calculate_angle_of_normal_vector_sec():
    assert Line(Vector(1, 1), Vector(1, 4)).calculate_angle_of_normal_vector() == 0

def test_mid_point():
    expected_output = "Vector(2.0, 2.0)"
    assert repr(Line(Vector(1, 1), Vector(3, 3)).mid_point()) == expected_output

def test_get_corner_coords():
    expected_output = "[(1, 1), (2, 1), (1, 2), (2, 2)]"
    assert repr(Square([Line(Vector(1, 1), Vector(2, 1)), Line(Vector(2, 1), Vector(2, 2)),
                        Line(Vector(2, 2), Vector(1, 2)), Line(Vector(1, 2), Vector(1, 1))]).get_corner_coords()) == expected_output

def test_is_point_inside():
    assert Square([Line(Vector(1, 1), Vector(2, 1)), Line(Vector(2, 1), Vector(2, 2)),
                   Line(Vector(2, 2), Vector(1, 2)), Line(Vector(1, 2), Vector(1, 1))]).is_point_inside(Vector(1.5, 1.5)) == True

def test_is_point_not_inside():
    assert Square([Line(Vector(1, 1), Vector(2, 1)), Line(Vector(2, 1), Vector(2, 2)),
                   Line(Vector(2, 2), Vector(1, 2)), Line(Vector(1, 2), Vector(1, 1))]).is_point_inside(Vector(3, 3)) == False

def test_points_inside_square():
    expected_output = [Vector(1.5, 1.5)]
    assert Square([Line(Vector(1, 1), Vector(2, 1)), Line(Vector(2, 1), Vector(2, 2)),
                   Line(Vector(2, 2), Vector(1, 2)), Line(Vector(1, 2), Vector(1, 1))]).points_inside_square([Vector(1.5, 1.5)]) == expected_output

def test_points_not_inside_square():
    assert Square([Line(Vector(1, 1), Vector(2, 1)), Line(Vector(2, 1), Vector(2, 2)),
                   Line(Vector(2, 2), Vector(1, 2)), Line(Vector(1, 2), Vector(1, 1))]).points_inside_square([Vector(3, 3)]) == []

def test_is_point_in_circle():
    assert Circle(Vector(3, 3), 3).is_point_in_circle(Vector(4, 4)) == True

def test_is_point_not_in_circle():
    assert Circle(Vector(3, 3), 3).is_point_in_circle(Vector(10, 10)) == False

def test_amount_of_points_in_circle():
    assert Circle(Vector(3, 3), 3).amount_of_points_in_circle([Vector(2, 2), Vector(4, 4)]) == 1





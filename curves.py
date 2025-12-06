#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# curves.py
# Copyright (C) 2016 xent
# Project is distributed under the terms of the GNU General Public License v3.0

import itertools
import math
import numpy as np

try:
    import model
except ImportError:
    from . import model


class Line:
    def __init__(self, beg, end, resolution):
        if resolution < 1:
            raise ValueError()
        self.beg = np.array(list(beg))
        self.end = np.array(list(end))
        self.resolution = resolution

    def apply(self, transform):
        self.beg = transform.apply(self.beg)
        self.end = transform.apply(self.end)

    def point(self, position):
        # Argument position is in range [0.0, 1.0]
        if 0.0 <= position <= 1.0:
            return self.beg * (1.0 - position) + self.end * position
        raise ValueError()

    def reverse(self):
        self.end, self.beg = self.beg, self.end

    def tessellate(self):
        scale = 1.0 / float(self.resolution)
        return [self.point(float(i) * scale) for i in range(0, self.resolution + 1)]


class Bezier(Line):
    def __init__(self, beg, beg_tension, end, end_tension, resolution):
        ''' Create Bezier curve.

        Bernstein polynomial of degree 3 is used:
            p0 is self.beg
            p1 is sum of self.beg and beg_tension
            p2 is sum of self.end and end_tension
            p3 is self.end
        '''
        super().__init__(beg, end, resolution)

        self.cbeg = self.beg + np.array(list(beg_tension))
        self.cend = self.end + np.array(list(end_tension))

    def apply(self, transform):
        self.cbeg = transform.apply(self.cbeg)
        self.cend = transform.apply(self.cend)
        super().apply(transform)

    def point(self, position):
        # Argument position is in range [0.0, 1.0]
        if 0.0 <= position <= 1.0:
            # Bernstein basis polynomials
            b03 = (1.0 - position) ** 3.0
            b13 = 3.0 * position * ((1.0 - position) ** 2.0)
            b23 = 3.0 * (position ** 2.0) * (1.0 - position)
            b33 = position ** 3.0
            return self.beg * b03 + self.cbeg * b13 + self.cend * b23 + self.end * b33
        raise ValueError()

    def reverse(self):
        self.cend, self.cbeg = self.cbeg, self.cend
        super().reverse()


class BezierQuad(model.Mesh):
    def __init__(self, a, b, c, d, resolution, inverse=False, points=None): # pylint: disable=invalid-name
        '''
        a[0] a[1] a[2] a[3]
        b[0] b[1] b[2] b[3]
        c[0] c[1] c[2] c[3]
        d[0] d[1] d[2] d[3]
        '''
        super().__init__()

        if resolution[0] < 1 or resolution[1] < 1:
            raise ValueError()

        # pylint: disable=invalid-name
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        # pylint: enable=invalid-name

        if points is None:
            steps = (1.0 / resolution[0], 1.0 / resolution[1])
            points = [
                [i * steps[0] for i in range(0, resolution[0] + 1)],
                [i * steps[1] for i in range(0, resolution[1] + 1)]
            ]

        self.tessellate(np.array(resolution) + 1, inverse, points)

    def interpolate(self, u, v): # pylint: disable=invalid-name
        def make_curve(row):
            return Bezier(row[0], row[1] - row[0], row[3], row[2] - row[3], 3)

        # pylint: disable=invalid-name
        a = make_curve(self.a)
        b = make_curve(self.b)
        c = make_curve(self.c)
        d = make_curve(self.d)
        q = make_curve((a.point(v), b.point(v), c.point(v), d.point(v)))
        # pylint: enable=invalid-name

        return q.point(u)

    def tessellate(self, resolution, inverse, points):
        total = resolution[0] * resolution[1]
        for j in range(0, resolution[1]):
            for i in range(0, resolution[0]):
                self.geo_vertices.append(self.interpolate(points[0][i], points[1][j]))

        for j in range(0, resolution[1] - 1):
            for i in range(0, resolution[0] - 1):
                # pylint: disable=invalid-name
                p1 = j * resolution[0] + i
                p2 = (p1 + 1) % total
                p3 = ((j + 1) * resolution[0] + i) % total
                p4 = (p3 + 1) % total
                # pylint: enable=invalid-name

                if inverse:
                    self.geo_polygons.append([p1, p3, p4, p2])
                else:
                    self.geo_polygons.append([p1, p2, p4, p3])


class BezierTri(model.Mesh):
    def __init__(self, a, b, c, mean, resolution, inverse=False): # pylint: disable=invalid-name
        '''
                    a[0]
                a[1]    a[2]
            b[2]    mean    c[1]
        b[0]    b[1]    c[2]    c[0]
        '''
        super().__init__()

        if resolution < 1:
            raise ValueError()

        # pylint: disable=invalid-name
        self.a = a
        self.b = b
        self.c = c
        # pylint: enable=invalid-name

        self.mean = mean

        self.tessellate(resolution, inverse)

    def interpolate(self, u, v, w): # pylint: disable=invalid-name
        return (
            self.a[0] * (u ** 3.0) + self.c[0] * (v ** 3.0) + self.b[0] * (w ** 3.0)
            + self.a[2] * 3.0 * v * (u ** 2.0) + self.a[1] * 3.0 * w * (u ** 2.0)
            + self.c[1] * 3.0 * u * (v ** 2.0) + self.c[2] * 3.0 * w * (v ** 2.0)
            + self.b[1] * 3.0 * v * (w ** 2.0) + self.b[2] * 3.0 * u * (w ** 2.0)
            + self.mean * 6.0 * u * v * w)

    def tessellate(self, resolution, inverse):
        row_offset = lambda row: sum(range(0, row + 1))
        point_u = np.array([1.0, 0.0, 0.0])
        point_v = np.array([0.0, 1.0, 0.0])
        point_w = np.array([0.0, 0.0, 1.0])

        self.geo_vertices.append(self.interpolate(*list(point_u)))

        for i in range(1, resolution + 1):
            v = (point_u * (resolution - i) + point_v * i) / resolution # pylint: disable=invalid-name
            w = (point_u * (resolution - i) + point_w * i) / resolution # pylint: disable=invalid-name

            for j in range(0, i + 1):
                u = (v * (i - j) + w * j) / i # pylint: disable=invalid-name
                self.geo_vertices.append(self.interpolate(*list(u)))

            if inverse:
                for j in range(0, i):
                    self.geo_polygons.append(
                        [row_offset(i) + j + 1, row_offset(i) + j, row_offset(i - 1) + j])
                for j in range(0, i - 1):
                    self.geo_polygons.append(
                        [row_offset(i - 1) + j, row_offset(i - 1) + j + 1, row_offset(i) + j + 1])
            else:
                for j in range(0, i):
                    self.geo_polygons.append(
                        [row_offset(i - 1) + j, row_offset(i) + j, row_offset(i) + j + 1])
                for j in range(0, i - 1):
                    self.geo_polygons.append(
                        [row_offset(i) + j + 1, row_offset(i - 1) + j + 1, row_offset(i - 1) + j])


def create_rotation_mesh(slices, wrap=True, inverse=False):
    geo_vertices = list(itertools.chain.from_iterable(slices))
    geo_polygons = []

    edges = len(slices) if wrap else len(slices) - 1
    size = len(slices[0])
    for i in range(0, edges):
        for vertex in range(0, size - 1):
            beg, end = i, i + 1 if i < len(slices) - 1 else 0
            if inverse:
                beg, end = end, beg

            beg_index, end_index = beg * size + vertex, end * size + vertex
            indices = [beg_index]
            if not model.Mesh.isclose(geo_vertices[beg_index], geo_vertices[end_index]):
                indices += [end_index]
            if not model.Mesh.isclose(geo_vertices[beg_index + 1], geo_vertices[end_index + 1]):
                indices += [end_index + 1]
            indices += [beg_index + 1]

            geo_polygons.append(indices)

    # Generate object
    mesh = model.Mesh()
    mesh.geo_vertices = geo_vertices
    mesh.geo_polygons = geo_polygons
    return mesh

def create_tri_cap_mesh(slices, inverse): # FIXME
    if inverse:
        vertices = [slices[i][0] for i in range(0, len(slices))]
    else:
        vertices = [slices[i][len(slices[i]) - 1] for i in range(0, len(slices))]

    indices = range(0, len(slices))
    geo_vertices = vertices + [sum(vertices) / len(slices)]
    geo_polygons = []

    if not inverse:
        for i, value in enumerate(indices):
            geo_polygons.append([len(vertices), value, indices[i - 1]])
    else:
        for i, value in enumerate(indices):
            geo_polygons.append([indices[i - 1], value, len(vertices)])

    # Generate object
    mesh = model.Mesh()
    mesh.geo_vertices = geo_vertices
    mesh.geo_polygons = geo_polygons
    return mesh

def loft(path, shape, translation=None, rotation=None, scaling=None, morphing=None):
    default_z_vec = np.array([0.0, 0.0, 1.0])

    if len(path) < 2:
        raise ValueError()
    if morphing is None:
        morphing = lambda _: shape
    if rotation is None:
        rotation = lambda _: np.zeros(3)
    if scaling is None:
        scaling = lambda _: np.ones(3)
    if translation is None:
        translation = lambda _: np.zeros(3)

    # Make initial rotation matrix
    path_vec = model.normalize(path[1][0:3] - path[0][0:3])
    rotation_vec = np.cross(default_z_vec, path_vec)
    if np.linalg.norm(rotation_vec) != 0.0:
        rotation_vec = model.normalize(rotation_vec)
        rotation_ang = math.acos(np.dot(default_z_vec, path_vec))
        matrix = model.make_rotation_matrix(rotation_vec, rotation_ang)
        previous_vec = path_vec
    else:
        matrix = np.identity(4)
        previous_vec = default_z_vec

    segments = []
    segments.append(model.Transform(matrix=matrix).quaternion())

    for i in range(1, len(path) - 1):
        path_vec = model.normalize(path[i + 1][0:3] - path[i][0:3])
        rotation_vec = np.cross(previous_vec, path_vec)
        if np.linalg.norm(rotation_vec) != 0.0:
            rotation_vec = model.normalize(rotation_vec)
            rotation_ang = math.acos(np.dot(previous_vec, path_vec))
            matrix = np.matmul(model.make_rotation_matrix(rotation_vec, rotation_ang), matrix)
            previous_vec = path_vec
        segments.append(model.Transform(matrix=matrix).quaternion())

    return make_loft_slices(path, segments, translation, rotation, scaling, morphing)

def make_loft_slices(path, segments, translation, rotation, scaling, morphing):
    slices = []

    count = len(segments)
    current = segments[0]

    for i in range(0, count + 1):
        if 0 < i < len(segments):
            quaternion = model.slerp(segments[i - 1], segments[i], 0.5)
            current = segments[i]
        else:
            quaternion = current

        shape = morphing(i)

        shape_transform = model.Transform()
        shape_transform.scale(scaling(i))
        shape_transform.translate(translation(i))
        transformed_shape = [shape_transform.apply(point) for point in shape]

        slice_transform = model.Transform(quaternion=quaternion)
        slice_transform.matrix = np.matmul(slice_transform.matrix,
                                              model.rpy_to_matrix(rotation(i)))
        slice_transform.translate(path[i])
        slices.append([slice_transform.apply(point) for point in transformed_shape])

    return slices

def optimize(points):
    if points:
        result = [points[0]]
        for point in points[1:]:
            if not model.Mesh.isclose(point, result[-1]):
                result.append(point)
        return result
    return []

def rotate(curve, axis, edges=None, angles=None):
    points = []
    for segment in curve:
        points.extend(segment.tessellate())
    points = optimize(points)
    slices = []

    if edges is not None and angles is None:
        angles = [(math.pi * 2.0 / edges) * i for i in range(0, edges)]
    elif edges is not None or angles is None:
        raise ValueError()

    for angle in angles:
        mat = model.make_rotation_matrix(axis, angle)
        slices.append([np.matmul(np.array([*p, 1.0]), mat)[0:3] for p in points])
    return slices

def calc_bezier_weight(a=None, b=None, angle=None): # pylint: disable=invalid-name
    if angle is None:
        if a is None or b is None:
            # User must provide vectors a and b when angle argument is not used
            raise TypeError()
        angle = model.angle(a, b)
    return (4.0 / 3.0) * math.tan(angle / 4.0)

def get_line_function(start, end):
    # Returns (A, B, C)
    delta_x, delta_y = end[0] - start[0], end[1] - start[1]

    if delta_x == 0.0:
        return (1.0, 0.0, -start[0])
    if delta_y == 0.0:
        return (0.0, 1.0, -start[1])
    return (delta_y, -delta_x, delta_x * start[1] - delta_y * start[0])

def intersect_line_plane(plane_point, plane_normal, line_start, line_end):
    line = model.normalize(line_end - line_start)
    if np.dot(plane_normal, line) == 0.0:
        return None
    line_length = np.linalg.norm(line_end - line_start)
    position = np.dot(plane_normal, plane_point - line_start) / np.dot(plane_normal, line)
    if position <= 0.0 or position >= line_length:
        return None
    return line * position + line_start

def intersect_line_functions(func_a, func_b, epsilon):
    # 2D lines are used, lines are described as (A, B, C) coefficients
    det = func_a[0] * func_b[1] - func_a[1] * func_b[0]
    if abs(det) <= epsilon:
        return None

    delta_x = -func_a[2] * func_b[1] + func_a[1] * func_b[2]
    delta_y = -func_a[0] * func_b[2] + func_a[2] * func_b[0]
    return (delta_x / det, delta_y / det)

def intersect_lines(line_a, line_b, epsilon=1e-6):
    # 2D lines are used, lines are described as (start, end) points
    line_a_func = get_line_function(line_a[0], line_a[1])
    line_b_func = get_line_function(line_b[0], line_b[1])

    cross_point = intersect_line_functions(line_a_func, line_b_func, epsilon)
    if cross_point is None:
        return None

    line_a_box = (
        (min(line_a[0][0], line_a[1][0]), min(line_a[0][1], line_a[1][1])),
        (max(line_a[0][0], line_a[1][0]), max(line_a[0][1], line_a[1][1]))
    )
    if not is_point_in_rect(line_a_box, cross_point, epsilon):
        return None

    line_b_box = (
        (min(line_b[0][0], line_b[1][0]), min(line_b[0][1], line_b[1][1])),
        (max(line_b[0][0], line_b[1][0]), max(line_b[0][1], line_b[1][1]))
    )
    if not is_point_in_rect(line_b_box, cross_point, epsilon):
        return None

    return cross_point

def is_point_in_rect(rect, point, epsilon=1e-6):
    in_range = lambda x, start, end: start - epsilon <= x <= end + epsilon
    if not in_range(point[0], rect[0][0], rect[1][0]):
        return False
    if not in_range(point[1], rect[0][1], rect[1][1]):
        return False
    return True

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# curves.py
# Copyright (C) 2016 xent
# Project is distributed under the terms of the GNU General Public License v3.0

import itertools
import math
import numpy

try:
    import model
except ImportError:
    from . import model


class Line:
    def __init__(self, beg, end, resolution):
        if resolution < 1:
            raise Exception()
        self.beg = numpy.array(list(beg))
        self.end = numpy.array(list(end))
        self.resolution = resolution

    def apply(self, transform):
        self.beg = transform.apply(self.beg)
        self.end = transform.apply(self.end)

    def point(self, position):
        # Argument position is in range [0.0, 1.0]
        if 0.0 <= position <= 1.0:
            return self.beg * (1.0 - position) + self.end * position
        raise Exception()

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

        self.cbeg = self.beg + numpy.array(list(beg_tension))
        self.cend = self.end + numpy.array(list(end_tension))

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
        raise Exception()

    def reverse(self):
        self.cend, self.cbeg = self.cbeg, self.cend
        super().reverse()


class BezierQuad(model.Mesh):
    def __init__(self, a, b, c, d, resolution, inverse=False): # pylint: disable=invalid-name
        '''
        a[0] a[1] a[2] a[3]
        b[0] b[1] b[2] b[3]
        c[0] c[1] c[2] c[3]
        d[0] d[1] d[2] d[3]
        '''
        super().__init__()

        if resolution[0] < 1 or resolution[1] < 1:
            raise Exception()

        # pylint: disable=invalid-name
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        # pylint: enable=invalid-name

        self.tessellate(numpy.array(resolution) + 1, inverse)

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

    def tessellate(self, resolution, inverse):
        step = ([1.0 / (resolution[0] - 1), 1.0 / (resolution[1] - 1)])
        total = resolution[0] * resolution[1]
        for j in range(0, resolution[1]):
            for i in range(0, resolution[0]):
                self.geo_vertices.append(self.interpolate(i * step[0], j * step[1]))

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
            raise Exception()

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
        point_u = numpy.array([1.0, 0.0, 0.0])
        point_v = numpy.array([0.0, 1.0, 0.0])
        point_w = numpy.array([0.0, 0.0, 1.0])

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


def optimize(points):
    if len(points) >= 1:
        result = [points[0]]
        for point in points[1:]:
            if not model.Mesh.isclose(point, result[-1]):
                result.append(point)
        return result
    return []

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
        slice_transform.matrix = numpy.matmul(slice_transform.matrix,
                                              model.rpy_to_matrix(rotation(i)))
        slice_transform.translate(path[i])
        slices.append([slice_transform.apply(point) for point in transformed_shape])

    return slices

def loft(path, shape, translation=None, rotation=None, scaling=None, morphing=None):
    default_z_vect = numpy.array([0.0, 0.0, 1.0])

    if len(path) < 2:
        raise Exception()
    if morphing is None:
        morphing = lambda _: shape
    if rotation is None:
        rotation = lambda _: numpy.zeros(3)
    if scaling is None:
        scaling = lambda _: numpy.ones(3)
    if translation is None:
        translation = lambda _: numpy.zeros(3)

    # Make initial rotation matrix
    z_vect = model.normalize(path[1][0:3] - path[0][0:3])
    x_vect = model.normalize(numpy.cross(default_z_vect, z_vect))
    if numpy.linalg.norm(x_vect) != 0.0:
        angle = math.acos(numpy.dot(default_z_vect, z_vect))
        matrix = model.make_rotation_matrix(x_vect, angle)
        previous_vect = z_vect
    else:
        matrix = numpy.identity(4)
        previous_vect = default_z_vect

    segments = []
    segments.append(model.Transform(matrix=matrix).quaternion())

    for i in range(1, len(path) - 1):
        z_vect = model.normalize(path[i + 1][0:3] - path[i][0:3])
        x_vect = model.normalize(numpy.cross(previous_vect, z_vect))
        if numpy.linalg.norm(x_vect) != 0.0:
            angle = math.acos(numpy.dot(previous_vect, z_vect))
            matrix = numpy.matmul(matrix, model.make_rotation_matrix(x_vect, angle))
            previous_vect = z_vect
        segments.append(model.Transform(matrix=matrix).quaternion())

    return make_loft_slices(path, segments, translation, rotation, scaling, morphing)

def rotate(curve, axis, edges=None, angles=None):
    points = []
    for segment in curve:
        points.extend(segment.tessellate())
    points = optimize(points)
    slices = []

    if edges is not None and angles is None:
        angles = [(math.pi * 2.0 / edges) * i for i in range(0, edges)]
    elif edges is not None or angles is None:
        raise Exception()

    for angle in angles:
        mat = model.make_rotation_matrix(axis, angle)
        slices.append([numpy.matmul(numpy.array([*p, 1.0]), mat)[0:3] for p in points])

    return slices

def create_tri_cap_mesh(slices, inverse): # FIXME
    if inverse:
        vertices = [slices[i][0] for i in range(0, len(slices))]
    else:
        vertices = [slices[i][len(slices[i]) - 1] for i in range(0, len(slices))]

    indices = range(0, len(slices))
    geo_vertices = vertices + [sum(vertices) / len(slices)]
    geo_polygons = []

    if not inverse:
        for i, _ in enumerate(indices):
            geo_polygons.append([len(vertices), indices[i], indices[i - 1]])
    else:
        for i, _ in enumerate(indices):
            geo_polygons.append([indices[i - 1], indices[i], len(vertices)])

    # Generate object
    mesh = model.Mesh()
    mesh.geo_vertices = geo_vertices
    mesh.geo_polygons = geo_polygons

    return mesh

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

def intersect_line_plane(plane_point, plane_normal, line_start, line_end):
    line = model.normalize(line_end - line_start)
    if numpy.dot(plane_normal, line) == 0.0:
        return None
    line_length = numpy.linalg.norm(line_end - line_start)
    position = numpy.dot(plane_normal, plane_point - line_start) / numpy.dot(plane_normal, line)
    if position <= 0.0 or position >= line_length:
        return None
    return line * position + line_start

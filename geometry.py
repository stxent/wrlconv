#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# geometry.py
# Copyright (C) 2015 xent
# Project is distributed under the terms of the GNU General Public License v3.0

import math
import numpy

try:
    import curves
    import model
except ImportError:
    from . import curves
    from . import model


class Box(model.Mesh):
    def __init__(self, size):
        super().__init__()

        half_size = numpy.array(size) / 2.0

        vertices = [
            numpy.array([ half_size[0],  half_size[1],  half_size[2]]),
            numpy.array([-half_size[0],  half_size[1],  half_size[2]]),
            numpy.array([-half_size[0], -half_size[1],  half_size[2]]),
            numpy.array([ half_size[0], -half_size[1],  half_size[2]]),
            numpy.array([ half_size[0],  half_size[1], -half_size[2]]),
            numpy.array([-half_size[0],  half_size[1], -half_size[2]]),
            numpy.array([-half_size[0], -half_size[1], -half_size[2]]),
            numpy.array([ half_size[0], -half_size[1], -half_size[2]])
        ]
        polygons = [
            [0, 1, 2, 3],
            [7, 6, 5, 4],
            [4, 5, 1, 0],
            [5, 6, 2, 1],
            [6, 7, 3, 2],
            [7, 4, 0, 3]
        ]

        self.geo_vertices = vertices
        self.geo_polygons = polygons


class Circle(model.Mesh):
    def __init__(self, radius, edges):
        if radius <= 0.0 or edges < 3:
            raise ValueError()
        super().__init__()

        angle, step = 0.0, math.pi * 2.0 / edges
        for i in range(0, edges):
            x, y = radius * math.cos(angle), radius * math.sin(angle) # pylint: disable=invalid-name
            self.geo_vertices.append(numpy.array([x, y, 0.0]))
            angle += step
        for i in range(1, edges - 1):
            self.geo_polygons.append([0, i, i + 1])


class Geosphere(model.Mesh):
    def __init__(self, radius, depth=1):
        if radius <= 0.0 or depth < 1:
            raise ValueError()
        super().__init__()

        r = (1.0 + math.sqrt(5.0)) / 4.0 # pylint: disable=invalid-name
        vertices = [
            numpy.array([-0.5,    r,  0.0]),
            numpy.array([ 0.5,    r,  0.0]),
            numpy.array([-0.5,   -r,  0.0]),
            numpy.array([ 0.5,   -r,  0.0]),
            numpy.array([ 0.0, -0.5,    r]),
            numpy.array([ 0.0,  0.5,    r]),
            numpy.array([ 0.0, -0.5,   -r]),
            numpy.array([ 0.0,  0.5,   -r]),
            numpy.array([   r,  0.0, -0.5]),
            numpy.array([   r,  0.0,  0.5]),
            numpy.array([  -r,  0.0, -0.5]),
            numpy.array([  -r,  0.0,  0.5])
        ]

        vertices = [model.normalize(vertex) * radius for vertex in vertices]
        polygons = [
            [ 0, 11,  5], [ 0,  5,  1], [ 0,  1,  7], [ 0,  7, 10], [ 0, 10, 11],
            [ 1,  5,  9], [ 5, 11,  4], [11, 10,  2], [10,  7,  6], [ 7,  1,  8],
            [ 3,  9,  4], [ 3,  4,  2], [ 3,  2,  6], [ 3,  6,  8], [ 3,  8,  9],
            [ 4,  9,  5], [ 2,  4, 11], [ 6,  2, 10], [ 8,  6,  7], [ 9,  8,  1]
        ]

        def get_middle_point(vect1, vect2):
            return model.normalize(vect1 + (vect2 - vect1) / 2.0) * radius

        for _ in range(0, depth):
            next_point = []
            for face in polygons:
                index = len(vertices)
                vertices.append(get_middle_point(vertices[face[0]], vertices[face[1]]))
                vertices.append(get_middle_point(vertices[face[1]], vertices[face[2]]))
                vertices.append(get_middle_point(vertices[face[2]], vertices[face[0]]))
                next_point.append([face[0], index + 0, index + 2])
                next_point.append([face[1], index + 1, index + 0])
                next_point.append([face[2], index + 2, index + 1])
                next_point.append([index + 0, index + 1, index + 2])
            polygons = next_point

        self.geo_vertices = vertices
        self.geo_polygons = polygons


class Plane(model.Mesh):
    def __init__(self, size, resolution):
        if size[0] <= 0.0 or size[1] <= 0.0 or resolution[0] < 1 or resolution[1] < 1:
            raise ValueError()
        super().__init__()

        res = (resolution[0] + 1, resolution[1] + 1)
        offset = (-size[0] / 2.0, -size[1] / 2.0)
        mult = size[0] / (res[0] - 1), size[1] / (res[1] - 1)
        total = res[0] * res[1]
        for j in range(0, res[1]):
            for i in range(0, res[0]):
                self.geo_vertices.append(
                    numpy.array([offset[0] + i * mult[0], offset[1] + j * mult[1], 0]))
        for j in range(0, res[1] - 1):
            for i in range(0, res[0] - 1):
                point1 = j * res[0] + i
                point2 = (point1 + 1) % total
                point3 = ((j + 1) * res[0] + i) % total
                point4 = (point3 + 1) % total
                self.geo_polygons.append([point1, point2, point4, point3])


def make_bezier_quad_outline(points, resolution=(1, 1), roundness=1.0 / 3.0):
    p01_vec = (points[1] - points[0]) * roundness
    p03_vec = (points[3] - points[0]) * roundness
    p21_vec = (points[1] - points[2]) * roundness
    p23_vec = (points[3] - points[2]) * roundness

    p10_vec = -p01_vec
    p12_vec = -p21_vec
    p30_vec = -p03_vec
    p32_vec = -p23_vec

    side_a = curves.Bezier(points[0], p01_vec, points[1], p10_vec, resolution[0])
    side_b = curves.Bezier(points[1], p12_vec, points[2], p21_vec, resolution[1])
    side_c = curves.Bezier(points[2], p23_vec, points[3], p32_vec, resolution[0])
    side_d = curves.Bezier(points[3], p30_vec, points[0], p03_vec, resolution[1])

    vertices = []
    vertices.extend(side_a.tessellate())
    vertices.extend(side_b.tessellate())
    vertices.extend(side_c.tessellate())
    vertices.extend(side_d.tessellate())
    return dict(zip(list(range(0, len(vertices))), vertices))

def make_circle_outline(center, radius, edges):
    vertices = []
    angle, delta = 0.0, math.pi * 2.0 / edges

    for _ in range(0, edges):
        # pylint: disable=invalid-name
        x = radius * math.cos(angle)
        y = radius * math.sin(angle)
        # pylint: enable=invalid-name

        vertices.append(center + numpy.array([x, y, 0.0]))
        angle += delta

    return dict(zip(list(range(0, len(vertices))), vertices))

def sort_vertices_by_angle(vertices, mean, normal, direction=None):
    keys = list(vertices.keys())
    if direction is None:
        direction = vertices[next(iter(vertices))] - mean
    angles = []
    for key in keys:
        vector = vertices[key] - mean
        angle = model.angle(direction, vector)
        if numpy.linalg.det(numpy.array([direction, vector, normal])) < 0.0:
            angle = -angle
        angles.append((key, angle))
    angles.sort(key=lambda x: x[1])
    return angles

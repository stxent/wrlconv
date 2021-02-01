#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# geometry.py
# Copyright (C) 2015 xent
# Project is distributed under the terms of the GNU General Public License v3.0

import math
import numpy

try:
    import model
except ImportError:
    from . import model


class Geosphere(model.Mesh):
    def __init__(self, radius, depth=1):
        if radius <= 0.0 or depth < 1:
            raise Exception()

        super().__init__()

        r = (1.0 + math.sqrt(5.0)) / 4.0 # pylint: disable=invalid-name
        vertices = []

        vertices.append(numpy.array([-0.5,    r,  0.0]))
        vertices.append(numpy.array([ 0.5,    r,  0.0]))
        vertices.append(numpy.array([-0.5,   -r,  0.0]))
        vertices.append(numpy.array([ 0.5,   -r,  0.0]))
        vertices.append(numpy.array([ 0.0, -0.5,    r]))
        vertices.append(numpy.array([ 0.0,  0.5,    r]))
        vertices.append(numpy.array([ 0.0, -0.5,   -r]))
        vertices.append(numpy.array([ 0.0,  0.5,   -r]))
        vertices.append(numpy.array([   r,  0.0, -0.5]))
        vertices.append(numpy.array([   r,  0.0,  0.5]))
        vertices.append(numpy.array([  -r,  0.0, -0.5]))
        vertices.append(numpy.array([  -r,  0.0,  0.5]))

        vertices = [model.normalize(v) * radius for v in vertices]
        polygons = []

        polygons.extend([[ 0, 11,  5], [ 0,  5,  1], [ 0,  1,  7], [ 0,  7, 10], [ 0, 10, 11]])
        polygons.extend([[ 1,  5,  9], [ 5, 11,  4], [11, 10,  2], [10,  7,  6], [ 7,  1,  8]])
        polygons.extend([[ 3,  9,  4], [ 3,  4,  2], [ 3,  2,  6], [ 3,  6,  8], [ 3,  8,  9]])
        polygons.extend([[ 4,  9,  5], [ 2,  4, 11], [ 6,  2, 10], [ 8,  6,  7], [ 9,  8,  1]])

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
            raise Exception()

        super().__init__()

        res = (resolution[0] + 1, resolution[1] + 1)
        offset = (-float(size[0]) / 2, -float(size[1]) / 2)
        mult = float(size[0]) / (res[0] - 1), float(size[1]) / (res[1] - 1)
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

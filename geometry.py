#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# geometry.py
# Copyright (C) 2015 xent
# Project is distributed under the terms of the GNU General Public License v3.0

import itertools
import math
import numpy as np

try:
    import curves
    import model
except ImportError:
    from . import curves
    from . import model


class Box(model.Mesh):
    def __init__(self, size):
        super().__init__()

        half_size = np.array(size) / 2.0

        vertices = [
            np.array([ half_size[0],  half_size[1],  half_size[2]]),
            np.array([-half_size[0],  half_size[1],  half_size[2]]),
            np.array([-half_size[0], -half_size[1],  half_size[2]]),
            np.array([ half_size[0], -half_size[1],  half_size[2]]),
            np.array([ half_size[0],  half_size[1], -half_size[2]]),
            np.array([-half_size[0],  half_size[1], -half_size[2]]),
            np.array([-half_size[0], -half_size[1], -half_size[2]]),
            np.array([ half_size[0], -half_size[1], -half_size[2]])
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
            self.geo_vertices.append(np.array([x, y, 0.0]))
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
            np.array([-0.5,    r,  0.0]),
            np.array([ 0.5,    r,  0.0]),
            np.array([-0.5,   -r,  0.0]),
            np.array([ 0.5,   -r,  0.0]),
            np.array([ 0.0, -0.5,    r]),
            np.array([ 0.0,  0.5,    r]),
            np.array([ 0.0, -0.5,   -r]),
            np.array([ 0.0,  0.5,   -r]),
            np.array([   r,  0.0, -0.5]),
            np.array([   r,  0.0,  0.5]),
            np.array([  -r,  0.0, -0.5]),
            np.array([  -r,  0.0,  0.5])
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
                    np.array([offset[0] + i * mult[0], offset[1] + j * mult[1], 0]))
        for j in range(0, res[1] - 1):
            for i in range(0, res[0] - 1):
                point1 = j * res[0] + i
                point2 = (point1 + 1) % total
                point3 = ((j + 1) * res[0] + i) % total
                point4 = (point3 + 1) % total
                self.geo_polygons.append([point1, point2, point4, point3])


def build_loft_mesh(slices, fill_start, fill_end):
    mesh = model.Mesh()

    number = len(slices[0])
    for points in slices:
        mesh.geo_vertices.extend(points)

    if fill_start:
        v_center_index = len(mesh.geo_vertices)
        mesh.geo_vertices.append(model.calc_median_point(slices[0]))

        for i in range(0, number - 1):
            mesh.geo_polygons.append([i, i + 1, v_center_index])
        if not model.Mesh.isclose(slices[0][0], slices[0][-1]):
            # Slice is not closed, append additional polygon
            mesh.geo_polygons.append([number - 1, 0, v_center_index])

    for i in range(0, len(slices) - 1):
        for j in range(0, number - 1):
            mesh.geo_polygons.append([
                i * number + j,
                (i + 1) * number + j,
                (i + 1) * number + j + 1,
                i * number + j + 1
            ])

    if fill_end:
        v_center_index = len(mesh.geo_vertices)
        v_start_index = (len(slices) - 1) * number
        mesh.geo_vertices.append(model.calc_median_point(slices[-1]))

        for i in range(v_start_index, v_start_index + number - 1):
            mesh.geo_polygons.append([i + 1, i, v_center_index])
        if not model.Mesh.isclose(slices[-1][0], slices[-1][-1]):
            # Slice is not closed, append additional polygon
            mesh.geo_polygons.append([v_start_index, v_start_index + number - 1, v_center_index])

    return mesh

def build_rotation_mesh(slices, wrap=True, inverse=False):
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

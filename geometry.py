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
        if radius <= 0. or depth < 1:
            raise Exception();

        model.Mesh.__init__(self)

        r = (1. + math.sqrt(5.)) / 4.
        vertices = []
        vertices.append(numpy.array([-.5,   r,  0.]))
        vertices.append(numpy.array([ .5,   r,  0.]))
        vertices.append(numpy.array([-.5,  -r,  0.]))
        vertices.append(numpy.array([ .5,  -r,  0.]))
        vertices.append(numpy.array([ 0., -.5,   r]))
        vertices.append(numpy.array([ 0.,  .5,   r]))
        vertices.append(numpy.array([ 0., -.5,  -r]))
        vertices.append(numpy.array([ 0.,  .5,  -r]))
        vertices.append(numpy.array([  r,  0., -.5]))
        vertices.append(numpy.array([  r,  0.,  .5]))
        vertices.append(numpy.array([ -r,  0., -.5]))
        vertices.append(numpy.array([ -r,  0.,  .5]))
        for i in range(0, len(vertices)):
            vertices[i] = model.normalize(vertices[i])
        polygons = []
        polygons.extend([[ 0, 11,  5], [ 0,  5,  1], [ 0,  1,  7], [ 0,  7, 10], [ 0, 10, 11]])
        polygons.extend([[ 1,  5,  9], [ 5, 11,  4], [11, 10,  2], [10,  7,  6], [ 7,  1,  8]])
        polygons.extend([[ 3,  9,  4], [ 3,  4,  2], [ 3,  2,  6], [ 3,  6,  8], [ 3,  8,  9]])
        polygons.extend([[ 4,  9,  5], [ 2,  4, 11], [ 6,  2, 10], [ 8,  6,  7], [ 9,  8,  1]])

        def getMiddlePoint(v1, v2):
            return model.normalize(v1 + (v2 - v1) / 2)

        for i in range(0, depth):
            pNext = []
            for face in polygons:
                index = len(vertices)
                vertices.append(getMiddlePoint(vertices[face[0]], vertices[face[1]]))
                vertices.append(getMiddlePoint(vertices[face[1]], vertices[face[2]]))
                vertices.append(getMiddlePoint(vertices[face[2]], vertices[face[0]]))
                pNext.append([face[0], index + 0, index + 2])
                pNext.append([face[1], index + 1, index + 0])
                pNext.append([face[2], index + 2, index + 1])
                pNext.append([index + 0, index + 1, index + 2])
            polygons = pNext

        self.geoVertices = vertices
        self.geoPolygons = polygons


class Plane(model.Mesh):
    def __init__(self, size, resolution):
        if size[0] <= 0. or size[1] <= 0. or resolution[0] < 1 or resolution[1] < 1:
            raise Exception();

        model.Mesh.__init__(self)

        res = (resolution[0] + 1, resolution[1] + 1)
        offset = (-float(size[0]) / 2, -float(size[1]) / 2)
        mult = float(size[0]) / (res[0] - 1), float(size[1]) / (res[1] - 1)
        total = res[0] * res[1]
        for y in range(0, res[1]):
            for x in range(0, res[0]):
                self.geoVertices.append(numpy.array([offset[0] + x * mult[0], offset[1] + y * mult[1], 0]))
        for y in range(0, res[1] - 1):
            for x in range(0, res[0] - 1):
                p1 = y * res[0] + x
                p2 = (p1 + 1) % total
                p3 = ((y + 1) * res[0] + x) % total
                p4 = (p3 + 1) % total
                self.geoPolygons.append([p1, p2, p4, p3])

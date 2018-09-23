#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# curves.py
# Copyright (C) 2016 xent
# Project is distributed under the terms of the GNU General Public License v3.0

import numpy
import math

try:
    import model
except ImportError:
    from . import model


class Line:
    def __init__(self, start, end, segments):
        if segments < 1:
            raise Exception()
        self.a = numpy.array(list(start))
        self.b = numpy.array(list(end))
        self.segments = segments

    def point(self, t):
        # Argument t is in range [0.0, 1.0]
        if t < 0.0 or t > 1.0:
            raise Exception()
        else:
            return self.a * (1.0 - t) + self.b * t

    def tesselate(self):
        scale = 1.0 / float(self.segments)
        points = []
        for i in range(0, self.segments + 1):
            points.append(self.point(float(i) * scale))
        return points


class Bezier(Line):
    def __init__(self, start, startTension, end, endTension, segments):
        Line.__init__(self, start, end, segments)
        self.ta = numpy.array(list(startTension))
        self.tb = numpy.array(list(endTension))

    def point(self, t):
        # Argument t is in range [0.0, 1.0]
        if t < 0.0 or t > 1.0:
            raise Exception()
        else:
            return self.a * math.pow(1.0 - t, 3) + (self.a + self.ta) * 3 * t * math.pow(1.0 - t, 2)\
                    + (self.b + self.tb) * 3 * math.pow(t, 2) * (1.0 - t) + self.b * math.pow(t, 3)


class BezierQuad(model.Mesh):
    def __init__(self, a, b, c, d, resolution, inverse=False):
        '''
        a[0] a[1] a[2] a[3]
        b[0] b[1] b[2] b[3]
        c[0] c[1] c[2] c[3]
        d[0] d[1] d[2] d[3]
        '''
        model.Mesh.__init__(self)

        if resolution[0] < 1 or resolution[1] < 1:
            raise Exception()

        self.a = a
        self.b = b
        self.c = c
        self.d = d

        self.tesselate(numpy.array(resolution) + 1, inverse)

    def interpolate(self, u, v):
        def makeCurve(row):
            return Bezier(row[0], row[1] - row[0], row[3], row[2] - row[3], 3)

        a = makeCurve(self.a)
        b = makeCurve(self.b)
        c = makeCurve(self.c)
        d = makeCurve(self.d)

        q = makeCurve((a.point(v), b.point(v), c.point(v), d.point(v)))
        return q.point(u)

    def tesselate(self, resolution, inverse):
        step = ([1.0 / (resolution[0] - 1), 1.0 / (resolution[1] - 1)])
        total = resolution[0] * resolution[1]
        for y in range(0, resolution[1]):
            for x in range(0, resolution[0]):
                self.geoVertices.append(self.interpolate(x * step[0], y * step[1]))

        for y in range(0, resolution[1] - 1):
            for x in range(0, resolution[0] - 1):
                p1 = y * resolution[0] + x
                p2 = (p1 + 1) % total
                p3 = ((y + 1) * resolution[0] + x) % total
                p4 = (p3 + 1) % total
                if inverse:
                    self.geoPolygons.append([p1, p3, p4, p2])
                else:
                    self.geoPolygons.append([p1, p2, p4, p3])


class BezierTriangle(model.Mesh):
    def __init__(self, a, b, c, mean, resolution, inverse=False):
        '''
                    a[0]
                a[1]    a[2]
            b[2]    mean    c[1]
        b[0]    b[1]    c[2]    c[0]
        '''
        model.Mesh.__init__(self)

        if resolution < 1:
            raise Exception()

        self.a = a
        self.b = b
        self.c = c
        self.mean = mean

        self.tesselate(resolution, inverse)

    def interpolate(self, u, v, w):
        return self.a[0] * math.pow(u, 3.0) + self.c[0] * math.pow(v, 3.0) + self.b[0] * math.pow(w, 3.0)\
                + self.a[2] * 3.0 * v * math.pow(u, 2.0) + self.a[1] * 3.0 * w * math.pow(u, 2.0)\
                + self.c[1] * 3.0 * u * math.pow(v, 2.0) + self.c[2] * 3.0 * w * math.pow(v, 2.0)\
                + self.b[1] * 3.0 * v * math.pow(w, 2.0) + self.b[2] * 3.0 * u * math.pow(w, 2.0)\
                + self.mean * 6.0 * u * v * w

    def tesselate(self, resolution, inverse):
        pu = numpy.array([1.0, 0.0, 0.0])
        pv = numpy.array([0.0, 1.0, 0.0])
        pw = numpy.array([0.0, 0.0, 1.0])

        self.geoVertices.append(self.interpolate(*list(pu)))

        def rowOffset(row):
            return sum(range(0, row + 1))

        for i in range(1, resolution + 1):
            v = (pu * (resolution - i) + pv * i) / resolution
            w = (pu * (resolution - i) + pw * i) / resolution

            for j in range(0, i + 1):
                u = (v * (i - j) + w * j) / i
                self.geoVertices.append(self.interpolate(*list(u)))

            if inverse:
                for j in range(0, i):
                    self.geoPolygons.append([rowOffset(i) + j + 1, rowOffset(i) + j, rowOffset(i - 1) + j])
                for j in range(0, i - 1):
                    self.geoPolygons.append([rowOffset(i - 1) + j, rowOffset(i - 1) + j + 1, rowOffset(i) + j + 1])
            else:
                for j in range(0, i):
                    self.geoPolygons.append([rowOffset(i - 1) + j, rowOffset(i) + j, rowOffset(i) + j + 1])
                for j in range(0, i - 1):
                    self.geoPolygons.append([rowOffset(i) + j + 1, rowOffset(i - 1) + j + 1, rowOffset(i - 1) + j])


def optimize(points):
    result = []
    for p in points:
        found = False
        for seen in result:
            if model.Mesh.comparePoints(p, seen):
                found = True
                break
        if not found:
            result.append(p)

    return result

def rotate(curve, axis, edges=None, angles=None):
    def pointToVector(p):
        return numpy.resize(numpy.append(p, [1.]), (4, 1))
    def vectorToPoint(v):
        return v.getA()[:,0][0:3]

    points = []
    [points.extend(element.tesselate()) for element in curve]
    points = optimize(points)
    slices = []

    if edges is not None and angles is None:
        angles = [(math.pi * 2. / float(edges)) * float(i) for i in range(0, edges)]
    elif edges is not None or angles is None:
        raise Exception()

    for angle in angles:
        mat = model.rotationMatrix(axis, angle)
        slices.append([vectorToPoint(mat * pointToVector(p)) for p in points])

    return slices

def createTriCapMesh(slices, beginning):
    if beginning:
        vertices = [slices[i][0] for i in range(0, len(slices))]
    else:
        vertices = [slices[i][len(slices[i]) - 1] for i in range(0, len(slices))]

    indices = range(0, len(slices))
    geoVertices = vertices + [sum(vertices) / len(slices)]
    geoPolygons = []

    if beginning:
        [geoPolygons.append([len(vertices), indices[i], indices[i - 1]]) for i in range(0, len(indices))]
    else:
        [geoPolygons.append([indices[i - 1], indices[i], len(vertices)]) for i in range(0, len(indices))]

    # Generate object
    mesh = model.Mesh()
    mesh.geoVertices = geoVertices
    mesh.geoPolygons = geoPolygons

    return mesh

def createRotationMesh(slices, wrap=True, inverse=False):
    geoVertices = []
    [geoVertices.extend(s) for s in slices]
    geoPolygons = []

    edges = len(slices) if wrap else len(slices) - 1
    size = len(slices[0])
    for i in range(0, edges):
        for vertex in range(0, size - 1):
            a, b = i, i + 1 if i < len(slices) - 1 else 0
            if inverse:
                a, b = b, a

            indices = []

            indices += [a * size + vertex]
            if not model.Mesh.comparePoints(geoVertices[a * size + vertex], geoVertices[b * size + vertex]):
                indices += [b * size + vertex]
            if not model.Mesh.comparePoints(geoVertices[a * size + vertex + 1], geoVertices[b * size + vertex + 1]):
                indices += [b * size + vertex + 1]
            indices += [a * size + vertex + 1]

            geoPolygons.append(indices)

    # Generate object
    mesh = model.Mesh()
    mesh.geoVertices = geoVertices
    mesh.geoPolygons = geoPolygons

    return mesh

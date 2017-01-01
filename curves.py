#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# curves.py
# Copyright (C) 2016 xent
# Project is distributed under the terms of the GNU General Public License v3.0

import numpy
import math

import model


class Line:
    def __init__(self, start, end, segments):
        if segments < 1:
            raise Exception()
        self.a = numpy.array(list(start))
        self.b = numpy.array(list(end))
        self.segments = segments

    def point(self, t):
        #Argument t is in range [0.0, 1.0]
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
        #Argument t is in range [0.0, 1.0]
        if t < 0.0 or t > 1.0:
            raise Exception()
        else:
            return self.a * math.pow(1.0 - t, 3) + (self.a + self.ta) * 3 * t * math.pow(1.0 - t, 2)\
                    + (self.b + self.tb) * 3 * math.pow(t, 2) * (1.0 - t) + self.b * math.pow(t, 3)


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

def rotate(curve, axis, edges):
    def pointToVector(p):
        return numpy.resize(numpy.append(p, [1.]), (4, 1))
    def vectorToPoint(v):
        return v.getA()[:,0][0:3]

    points = []
    [points.extend(element.tesselate()) for element in curve]
    points = optimize(points)
    slices = []

    for i in range(0, edges):
        mat = model.rotationMatrix(axis, (math.pi * 2. / float(edges)) * float(i))
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

    #Generate object
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
            geoPolygons.append([
                    a * size + vertex,
                    b * size + vertex,
                    b * size + vertex + 1,
                    a * size + vertex + 1
            ])

    #Generate object
    mesh = model.Mesh()
    mesh.geoVertices = geoVertices
    mesh.geoPolygons = geoPolygons

    return mesh

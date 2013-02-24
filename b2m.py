#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: xent (alexdmitv@gmail.com)
# License: Public domain code
# Version: 0.3b
import re
import math
import numpy
import sys
import time
import argparse
import os
import subprocess
import Image, ImageDraw

class Rect:
    class RectCorner:
        def __init__(self, position, chamfer):
            self.position = position #Tuple of two elements: x and y
            self.chamfer = chamfer #Tuple of two elements: x and y chamfers

        def generate(self, vect):
            vertices = []
            vertices.append([self.position[0] + self.chamfer[0] * vect[0],
                    self.position[1]])
            vertices.append([self.position[0] + self.chamfer[0] * vect[0],
                    self.position[1] + self.chamfer[1] * vect[1]])
            vertices.append([self.position[0],
                    self.position[1] + self.chamfer[1] * vect[1]])
            return vertices

    def __init__(self, points, chamfers):
        cpoints = [[points[0][0], points[0][1]],
                   [points[1][0], points[0][1]],
                   [points[1][0], points[1][1]],
                   [points[0][0], points[1][1]]]
        self.coords = points #Coordinates of top left and bottom right corners
        self.corners = []
        self.sub = None
        for i in range(0, 4):
            self.corners.append(Rect.RectCorner(cpoints[i], chamfers[i]))

    def contain(self, points):
        if self.coords[0][0] <= points[0][0] and points[1][0] <= self.coords[1][0] and \
           self.coords[0][1] <= points[0][1] and points[1][1] <= self.coords[1][1]:
            return True
        else:
            return False

    def intersect(self, points):
        #Returns intersected edge
        #Horizontal edges
        if self.coords[0][0] <= points[0][0] and points[1][0] <= self.coords[1][0]:
            if points[0][1] <= self.coords[0][1] <= points[1][1]:
                return (0, 1) #Top
            if points[0][1] <= self.coords[1][1] <= points[1][1]:
                return (2, 3) #Bottom
        #Vertical edges
        if self.coords[0][1] <= points[0][1] and points[1][1] <= self.coords[1][1]:
            if points[0][0] <= self.coords[0][0] <= points[1][0]:
                return (3, 0) #Left
            if points[0][0] <= self.coords[1][0] <= points[1][0]:
                return (1, 2) #Right
        return None

    def tesselate(self):
        #Returns tuple with vertex and polygon lists
        if self.sub is None:
            vertices = []
            polygons = [[ 0,  1,  4,  3],
                        [ 5,  4,  7,  8],
                        [ 7,  6,  9, 10],
                        [10, 11,  2,  1],
                        [ 1,  4,  7, 10]]
            vertices.extend(self.corners[0].generate(( 1,  1)))
            vertices.extend(self.corners[1].generate((-1,  1)))
            vertices.extend(self.corners[2].generate((-1, -1)))
            vertices.extend(self.corners[3].generate(( 1, -1)))
            return (vertices, polygons)
        else:
            vertices = []
            polygons = []
            for entry in self.sub:
                (vList, pList) = entry.tesselate()
                for poly in pList:
                    for i in range(0, len(poly)):
                        poly[i] += len(vertices)
                vertices.extend(vList)
                polygons.extend(pList)
            return (vertices, polygons)

    def subdivide(self, points):
        size = (points[1][0] - points[0][0], points[1][1] - points[0][1])
        center = (points[0][0] + size[0] / 2, points[0][1] + size[1] / 2)
        if self.sub is None:
            if self.contain(points):
                self.sub = []
                self.sub.append(Rect(((self.coords[0][0], self.coords[0][1]), center),
                        (self.corners[0].chamfer, (0, 0), (size[0] / 2, size[1] / 2), (0, 0))))
                self.sub.append(Rect(((center[0], self.coords[0][1]), (self.coords[1][0], center[1])),
                        ((0, 0), self.corners[1].chamfer, (0, 0), (size[0] / 2, size[1] / 2))))
                self.sub.append(Rect((center, (self.coords[1][0], self.coords[1][1])),
                        ((size[0] / 2, size[1] / 2), (0, 0), self.corners[2].chamfer, (0, 0))))
                self.sub.append(Rect(((self.coords[0][0], center[1]), (center[0], self.coords[1][1])),
                        ((0, 0), (size[0] / 2, size[1] / 2), (0, 0), self.corners[3].chamfer)))
        else:
            for entry in self.sub:
                entry.subdivide(points)

test = Rect(((100, 100), (700, 700)), ((10, 15), (20, 10), (15, 20), (30, 40)))
#test = Rect(((100, 100), (200, 200)), ((20, 20), (20, 20), (20, 20), (20, 20)))

sizeX, sizeY = 800, 800
colorData = Image.new("RGB", (sizeX, sizeY))
drawing = ImageDraw.Draw(colorData)

test.subdivide(((300, 200), (340, 220)))
test.subdivide(((600, 600), (640, 640)))
test.subdivide(((400, 400), (420, 420)))
test.subdivide(((150, 400), (170, 420)))

(vert, poly) = test.tesselate()
for i in range(0, len(poly)):
    p = poly[i]
    drawing.line([(vert[p[0]][0], vert[p[0]][1]), (vert[p[1]][0], vert[p[1]][1])])
    drawing.line([(vert[p[1]][0], vert[p[1]][1]), (vert[p[2]][0], vert[p[2]][1])])
    drawing.line([(vert[p[2]][0], vert[p[2]][1]), (vert[p[3]][0], vert[p[3]][1])])
    drawing.line([(vert[p[3]][0], vert[p[3]][1]), (vert[p[0]][0], vert[p[0]][1])])

col = ((255, 0, 0), (0, 255, 0), (0, 0, 255))
for i in range(0, len(vert)):
    v = vert[i]
    drawing.line([(v[0] - 2, v[1]), (v[0] + 2, v[1])], col[i % 3])
    drawing.line([(v[0], v[1] - 2), (v[0], v[1] + 2)], col[i % 3])

colorData.show()

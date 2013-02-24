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
        self.corners = []
        for i in range(0, 4):
            self.corners.append(Rect.RectCorner(cpoints[i], chamfers[i]))

    def tesselate(self):
        #Returns tuple with vertex and polygon lists
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


test = Rect(((100, 100), (200, 200)), ((10, 15), (20, 10), (7, 14), (30, 40)))
#test = Rect(((100, 100), (200, 200)), ((20, 20), (20, 20), (20, 20), (20, 20)))

sizeX, sizeY = 800, 800
colorData = Image.new("RGB", (sizeX, sizeY))
drawing = ImageDraw.Draw(colorData)

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

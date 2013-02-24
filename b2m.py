#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: xent (alexdmitv@gmail.com)
# License: Public domain code
# Version: 0.3b
import numpy
import Image, ImageDraw
import random
#import re
#import math
#import sys
#import time
#import argparse
#import os
#import subprocess

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

    @staticmethod
    def prCollision(coords, point):
        top, bottom = coords[0], coords[1]
        return top[0] <= point[0] <= bottom[0] and top[1] <= point[1] <= bottom[1]

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
        return Rect.prCollision(self.coords, points[0]) and Rect.prCollision(self.coords, points[1])

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
        #Top left
        if Rect.prCollision(points, self.coords[0]):
            return (0)
        #Top right
        if Rect.prCollision(points, (self.coords[1][0], self.coords[0][1])):
            return (1)
        #Bottom left
        if Rect.prCollision(points, self.coords[1]):
            return (2)
        #Bottom right
        if Rect.prCollision(points, (self.coords[0][0], self.coords[1][1])):
            return (3)
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
                #Top left
                coords = (self.coords[0], center)
                chamfers = (self.corners[0].chamfer, (0, 0), (size[0] / 2, size[1] / 2), (0, 0))
                self.sub.append(Rect(coords, chamfers))
                #Top right
                coords = ((center[0], self.coords[0][1]), (self.coords[1][0], center[1]))
                chamfers = ((0, 0), self.corners[1].chamfer, (0, 0), (size[0] / 2, size[1] / 2))
                self.sub.append(Rect(coords, chamfers))
                #Bottom right
                coords = (center, self.coords[1])
                chamfers = ((size[0] / 2, size[1] / 2), (0, 0), self.corners[2].chamfer, (0, 0))
                self.sub.append(Rect(coords, chamfers))
                #Bottom left
                coords = ((self.coords[0][0], center[1]), (center[0], self.coords[1][1]))
                chamfers = ((0, 0), (size[0] / 2, size[1] / 2), (0, 0), self.corners[3].chamfer)
                self.sub.append(Rect(coords, chamfers))
            else:
                edge = self.intersect(points)
                if edge in ((0, 1), (2, 3)):
                    self.sub = []
                    if edge == (0, 1):
                        topChamfer = (size[0] / 2, points[1][1] - self.coords[0][1])
                        bottomChamfer = (0, 0)
                    else:
                        topChamfer = (0, 0)
                        bottomChamfer = (size[0] / 2, self.coords[1][1] - points[0][1])
                    #Left
                    coords = (self.coords[0], (center[0], self.coords[1][1]))
                    chamfers = (self.corners[0].chamfer, topChamfer, bottomChamfer, self.corners[3].chamfer)
                    self.sub.append(Rect(coords, chamfers))
                    #Right
                    coords = ((center[0], self.coords[0][1]), self.coords[1])
                    chamfers = (topChamfer, self.corners[1].chamfer, self.corners[2].chamfer, bottomChamfer)
                    self.sub.append(Rect(coords, chamfers))
                if edge in ((1, 2), (3, 0)):
                    self.sub = []
                    if edge == (3, 0):
                        leftChamfer = (points[1][0] - self.coords[0][0], size[1] / 2)
                        rightChamfer = (0, 0)
                    else:
                        leftChamfer = (0, 0)
                        rightChamfer = (self.coords[1][0] - points[0][0], size[1] / 2)
                    #Top
                    coords = (self.coords[0], (self.coords[1][0], center[1]))
                    chamfers = (self.corners[0].chamfer, self.corners[1].chamfer, rightChamfer, leftChamfer)
                    self.sub.append(Rect(coords, chamfers))
                    #Bottom
                    coords = ((self.coords[0][0], center[1]), self.coords[1])
                    chamfers = (leftChamfer, rightChamfer, self.corners[2].chamfer, self.corners[3].chamfer)
                    self.sub.append(Rect(coords, chamfers))
                if edge in ((0), (1), (2), (3)):
                    pointToCorner = {(0): 0, (1): 1, (2): 2, (3): 3}
                    corner = pointToCorner[edge]
                    if edge == (0):
                        self.corners[corner].chamfer = (points[1][0] - self.coords[0][0],
                                points[1][1] - self.coords[0][1])
                    elif edge == (1):
                        self.corners[corner].chamfer = (self.coords[1][0] - points[0][0],
                                points[1][1] - self.coords[0][1])
                    elif edge == (2):
                        self.corners[corner].chamfer = (self.coords[1][0] - points[0][0],
                                self.coords[1][1] - points[0][1])
                    elif edge == (3):
                        self.corners[corner].chamfer = (points[1][0] - self.coords[0][0],
                                self.coords[1][1] - points[0][1])
        else:
            for entry in self.sub:
                entry.subdivide(points)


def circleRect(position, radius):
    return ((position[0] - radius, position[1] - radius), (position[0] + radius, position[1] + radius))

def writeVRML(filename, vertices, polygons):
    offset, scale = (-400, -400), (0.025, 0.025)
    out = open(filename, "wb")
    out.write("#VRML V2.0 utf8\n#Created by b2m.py\n")
    out.write("DEF OB_Board Transform {\n")
    out.write("\ttranslation 0 0 0\n")
    out.write("\trotation 1 0 0 0\n")
    out.write("\tscale 1 1 1\n")
    out.write("\tchildren [\n"
              "\t\tDEF ME_Cube Group {\n"
              "\t\t\tchildren [\n"
              "\t\t\t\tShape {\n")
    out.write("\t\t\t\t\tappearance Appearance {\n"
              "\t\t\t\t\t\tmaterial DEF lapp Material {\n"
              "\t\t\t\t\t\t\tdiffuseColor 0.1939154 0.7799909 0.5564991\n"
              "\t\t\t\t\t\t\tambientIntensity 0.3333333\n"
              "\t\t\t\t\t\t\tspecularColor 0.4012008 0.8012008 0.4012008\n"
              "\t\t\t\t\t\t\temissiveColor  0.0 0.0 0.0\n"
              "\t\t\t\t\t\t\tshininess 0.95\n"
              "\t\t\t\t\t\t\ttransparency 0.0\n"
              "\t\t\t\t\t\t}\n"
              "\t\t\t\t\t}\n")
    out.write("\t\t\t\t\tgeometry IndexedFaceSet {\n"
              "\t\t\t\t\t\tsolid FALSE\n"
              "\t\t\t\t\t\tcoord DEF coord_Cube Coordinate {\n"
              "\t\t\t\t\t\t\tpoint [\n")
    for v in vertices:
        out.write("%f %f %f\n" % ((v[0] + offset[0]) * scale[0], (v[1] + offset[1]) * scale[1], 0.0))
    out.write("\t\t\t\t\t\t\t]\n"
              "\t\t\t\t\t\t}\n"
              "\t\t\t\t\t\tcoordIndex [\n")
    for p in polygons:
        for index in p:
            out.write("%u " % index)
        out.write("-1,\n")
    out.write("\t\t\t\t\t\t]\n"
              "\t\t\t\t\t}\n"
              "\t\t\t\t}\n"
              "\t\t\t]\n"
              "\t\t}\n"
              "\t]\n"
              "}\n")
    out.close()

test = Rect(((50, 50), (750, 750)), ((10, 15), (20, 10), (15, 20), (20, 30)))

random.seed()
holes = []
for i in range(0, 4):
    pos = (random.randint(100, 700), random.randint(100, 700))
    rad = random.randint(5, 15)
    holes.append((pos, rad))

for h in holes:
    test.subdivide(circleRect(h[0], h[1]))

(vert, poly) = test.tesselate()
print "Complexity: vertices %u, polygons %u" % (len(vert), len(poly))

sizeX, sizeY = 800, 800
colorData = Image.new("RGB", (sizeX, sizeY))
drawing = ImageDraw.Draw(colorData)
col = ((255, 0, 0), (0, 255, 0), (255, 255, 0))

for i in range(0, len(poly)):
    p = poly[i]
    drawing.line([(vert[p[0]][0], vert[p[0]][1]), (vert[p[1]][0], vert[p[1]][1])], (128, 128, 128))
    drawing.line([(vert[p[1]][0], vert[p[1]][1]), (vert[p[2]][0], vert[p[2]][1])], (128, 128, 128))
    drawing.line([(vert[p[2]][0], vert[p[2]][1]), (vert[p[3]][0], vert[p[3]][1])], (128, 128, 128))
    drawing.line([(vert[p[3]][0], vert[p[3]][1]), (vert[p[0]][0], vert[p[0]][1])], (128, 128, 128))
for i in range(0, len(vert)):
    v = vert[i]
    drawing.line([(v[0] - 2, v[1]), (v[0] + 2, v[1])], col[i % 3])
    drawing.line([(v[0], v[1] - 2), (v[0], v[1] + 2)], col[i % 3])

writeVRML("board.wrl", vert, poly)
colorData.show()

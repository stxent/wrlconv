#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: xent (alexdmitv@gmail.com)
# License: Public domain code
# Version: 0.3b
import numpy
import Image, ImageDraw
import random
import copy
import math
#import re
#import sys
#import time
#import argparse
#import os
#import subprocess

class Rect:
    THICKNESS = 0.25
    class RectCorner:
        def __init__(self, position, chamfer):
            self.position = position #Tuple of two elements: x and y
            self.chamfer = chamfer #Tuple of two elements: x and y chamfers

        def generate(self, vect):
            vertices = []
            vertices.append([self.position[0] + self.chamfer[0] * vect[0],
                    self.position[1], Rect.THICKNESS])
            vertices.append([self.position[0] + self.chamfer[0] * vect[0],
                    self.position[1] + self.chamfer[1] * vect[1], Rect.THICKNESS])
            vertices.append([self.position[0],
                    self.position[1] + self.chamfer[1] * vect[1], Rect.THICKNESS])
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
            polygons = [[ 1,  0,  3,  4],
                        [ 4,  5,  8,  7],
                        [ 7,  6,  9, 10],
                        [10, 11,  2,  1],
                        [ 1,  4,  7, 10]]
            vertices.extend(self.corners[0].generate(( 1,  1)))
            vertices.extend(self.corners[1].generate((-1,  1)))
            vertices.extend(self.corners[2].generate((-1, -1)))
            vertices.extend(self.corners[3].generate(( 1, -1)))

            rebuilded = []
            vIndex = range(0, len(vertices))
            while len(vIndex):
                vert = vertices[vIndex[0]]
                same = []
                for i in range(0, len(vertices)):
                    if vertices[i] == vert:
                        same.append(i)
                last = len(rebuilded)
                for poly in polygons:
                    for i in range(0, len(poly)):
                        if poly[i] in same:
                            poly[i] = last
                for ind in same:
                    vIndex.remove(ind)
                rebuilded.append(vert)
            for i in range(len(polygons) - 1, -1, -1):
                failed = False
                for j in range(len(polygons[i]) - 1, -1, -1):
                    if polygons[i].count(polygons[i][j]) > 1:
                        polygons[i].pop(j)
                if len(polygons[i]) < 3:
                    polygons.pop(i)
            return (rebuilded, polygons)
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

    def borders(self):
        if self.sub is None:
            vertices = []
            polygons = [[ 2, 11, 23, 14],
                        [ 0,  3, 15, 12],
                        [ 5,  8, 20, 17],
                        [ 6,  9, 21, 18]]
            vertices.extend(self.corners[0].generate(( 1,  1)))
            vertices.extend(self.corners[1].generate((-1,  1)))
            vertices.extend(self.corners[2].generate((-1, -1)))
            vertices.extend(self.corners[3].generate(( 1, -1)))
            for vert in vertices:
                vert[2] = -Rect.THICKNESS
            vertices.extend(self.corners[0].generate(( 1,  1)))
            vertices.extend(self.corners[1].generate((-1,  1)))
            vertices.extend(self.corners[2].generate((-1, -1)))
            vertices.extend(self.corners[3].generate(( 1, -1)))
            return (vertices, polygons)
        else:
            return ([], [])

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


def optimizeVertices(vertices, polygons):
    retVert = []
    retPoly = copy.deepcopy(polygons)
    vIndex = range(0, len(vertices))
    while len(vIndex):
        vert = vertices[vIndex[0]]
        same = []
        for i in range(0, len(vertices)):
            if vertices[i] == vert:
                same.append(i)
        last = len(retVert)
        for poly in retPoly:
            for i in range(0, len(poly)):
                if poly[i] in same:
                    poly[i] = last
        for ind in same:
            vIndex.remove(ind)
        retVert.append(vert)
    return (retVert, retPoly)

def circleRect(position, radius):
    return ((position[0] - radius, position[1] - radius), (position[0] + radius, position[1] + radius))

def createBoard(vertices, polygons):
    volVertices = []
    volPolygons = []
    for vert in vertices:
        volVertices.append(vert)
    offset = len(volVertices)
    for vert in vertices:
        volVertices.append([vert[0], vert[1], -0.25])
    for poly in polygons:
        newPoly = []
        for i in range(0, len(poly)):
            newPoly.append(poly[i])
        volPolygons.append(newPoly)
        newPoly = []
        for i in range(len(poly) - 1, -1, -1):
            newPoly.append(poly[i] + offset)
        volPolygons.append(newPoly)
    return (volVertices, volPolygons)

def createHole(position, radius):
    edges = 24
    vertices = []
    polygons = []
    delta = math.pi * 2 / 24
    angle = 0
    for i in range(0, edges):
        vertices.append([position[0] + radius * math.cos(angle), position[1] + radius * math.sin(angle), 0.25])
        vertices.append([position[0] + radius * math.cos(angle), position[1] + radius * math.sin(angle), -0.25])
        angle += delta
    polygons.append([(edges - 1) * 2, 0, 1, (edges - 1) * 2 + 1])
    for i in range(0, edges - 1):
        polygons.append([(i + 0) * 2, (i + 1) * 2, (i + 1) * 2 + 1, (i + 0) * 2 + 1])

    vertices.append([position[0] + radius, position[1] + radius, 0.25])
    vertices.append([position[0] + radius, position[1] + radius, -0.25])
    vertices.append([position[0] - radius, position[1] + radius, 0.25])
    vertices.append([position[0] - radius, position[1] + radius, -0.25])
    vertices.append([position[0] - radius, position[1] - radius, 0.25])
    vertices.append([position[0] - radius, position[1] - radius, -0.25])
    vertices.append([position[0] + radius, position[1] - radius, 0.25])
    vertices.append([position[0] + radius, position[1] - radius, -0.25])

    offset = edges * 2
    delta = edges / 4
    for i in range(0, delta):
        #Front
        polygons.append([(i + 0) * 2 + delta * 2 * 0, offset + 0, (i + 1) * 2 + delta * 2 * 0])
        polygons.append([(i + 0) * 2 + delta * 2 * 1, offset + 2, (i + 1) * 2 + delta * 2 * 1])
        polygons.append([(i + 0) * 2 + delta * 2 * 2, offset + 4, (i + 1) * 2 + delta * 2 * 2])
        polygons.append([(i + 0) * 2 + delta * 2 * 3, offset + 6, (i + 1) * 2 + delta * 2 * 3])
        #Back
        polygons.append([(i + 1) * 2 + delta * 2 * 0 + 1, offset + 1, (i + 0) * 2 + delta * 2 * 0 + 1])
        polygons.append([(i + 1) * 2 + delta * 2 * 1 + 1, offset + 3, (i + 0) * 2 + delta * 2 * 1 + 1])
        polygons.append([(i + 1) * 2 + delta * 2 * 2 + 1, offset + 5, (i + 0) * 2 + delta * 2 * 2 + 1])
        polygons.append([(i + 1) * 2 + delta * 2 * 3 + 1, offset + 7, (i + 0) * 2 + delta * 2 * 3 + 1])

    return (vertices, polygons)

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
        out.write("%f %f %f\n" % ((v[0] + offset[0]) * scale[0], (v[1] + offset[1]) * scale[1], v[2]))
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

#test = Rect(((50, 50), (750, 750)), ((10, 15), (20, 10), (15, 20), (20, 30)))
#test = Rect(((50, 50), (750, 750)), ((50, 50), (50, 50), (50, 50), (50, 50)))
test = Rect(((50, 50), (750, 750)), ((0, 0), (0, 0), (0, 0), (0, 0)))
bVert, bPoly = test.borders()

random.seed()
holes = []
holes.append(((400, 400), 50))
holes.append(((500, 350), 30))
#for i in range(0, 2):
    #pos = (random.randint(100, 700), random.randint(100, 700))
    #rad = random.randint(20, 50)
    #holes.append((pos, rad))

for h in holes:
    test.subdivide(circleRect(h[0], h[1]))

vert, poly = test.tesselate()
print "Complexity: vertices %u, polygons %u" % (len(vert), len(poly))
vert, poly = optimizeVertices(vert, poly)
print "Complexity: vertices %u, polygons %u" % (len(vert), len(poly))

sizeX, sizeY = 800, 800
colorData = Image.new("RGB", (sizeX, sizeY))
drawing = ImageDraw.Draw(colorData)
col = ((255, 0, 0), (0, 255, 0), (255, 255, 0))

for i in range(0, len(poly)):
    for j in range(0, len(poly[i]) - 1):
        va, vb = poly[i][j], poly[i][j + 1]
        drawing.line([(vert[va][0], vert[va][1]), (vert[vb][0], vert[vb][1])], (128, 128, 128))
    va, vb = poly[i][len(poly[i]) - 1], poly[i][0]
    drawing.line([(vert[va][0], vert[va][1]), (vert[vb][0], vert[vb][1])], (128, 128, 128))
for i in range(0, len(vert)):
    v = vert[i]
    drawing.line([(v[0] - 2, v[1]), (v[0] + 2, v[1])], col[i % 3])
    drawing.line([(v[0], v[1] - 2), (v[0], v[1] + 2)], col[i % 3])

xVert, xPoly = createBoard(vert, poly)
for poly in bPoly:
    for i in range(0, len(poly)):
        poly[i] += len(xVert)
xVert.extend(bVert)
xPoly.extend(bPoly)

print "Complexity: vertices %u, polygons %u" % (len(xVert), len(xPoly))
for h in holes:
    hVert, hPoly = createHole(h[0], h[1])
    for poly in hPoly:
        for i in range(0, len(poly)):
            poly[i] += len(xVert)
    xVert.extend(hVert)
    xPoly.extend(hPoly)

print "Complexity: vertices %u, polygons %u" % (len(xVert), len(xPoly))
#xVert, xPoly = optimizeVertices(xVert, xPoly)
#print "Complexity: vertices %u, polygons %u" % (len(xVert), len(xPoly))

writeVRML("board.wrl", xVert, xPoly)
colorData.show()

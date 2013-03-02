#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# b2m.py
# Copyright (C) 2013 xent
# Project is distributed under the terms of the GNU General Public License v3.0

import argparse
import copy
import numpy
import math
import os
import random
import re

import Image
import ImageDraw

import model
#import sys
#import time
#import subprocess

class Rect:
    THICKNESS = 0.2
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

    @staticmethod
    def rCollision(ca, cb):
        return (ca[0][0] <= cb[0][0] <= ca[1][0] or cb[0][0] <= ca[0][0] <= cb[1][0]) and \
               (ca[0][1] <= cb[0][1] <= ca[1][1] or cb[0][1] <= ca[0][1] <= cb[1][1])

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
            polygons = []
            vertices.extend(self.corners[0].generate(( 1,  1)))
            vertices.extend(self.corners[1].generate((-1,  1)))
            vertices.extend(self.corners[2].generate((-1, -1)))
            vertices.extend(self.corners[3].generate(( 1, -1)))
            amorph0 = self.coords[1][1] - self.corners[2].chamfer[1] < self.coords[0][1] + self.corners[0].chamfer[1]
            amorph1 = self.coords[1][1] - self.corners[3].chamfer[1] < self.coords[0][1] + self.corners[1].chamfer[1]
            amorph2 = self.coords[1][0] - self.corners[2].chamfer[0] < self.coords[0][0] + self.corners[0].chamfer[0]
            amorph3 = self.coords[1][0] - self.corners[3].chamfer[0] < self.coords[0][0] + self.corners[1].chamfer[0]
            if amorph0 or amorph1:
                polygons.append([ 1, 10, 11,  2])
                polygons.append([ 4,  5,  8,  7])
                if amorph0:
                    polygons.append([ 0,  3,  4,  7])
                    polygons.append([ 0,  7,  6,  1])
                    polygons.append([ 1,  6,  9, 10])
                else:
                    polygons.append([ 0,  3, 10,  1])
                    polygons.append([ 3,  4,  9, 10])
                    polygons.append([ 4,  7,  6,  9])
            elif amorph2 or amorph3:
                polygons.append([ 7,  6,  9, 10])
                polygons.append([ 0,  3,  4,  1])
                if amorph2:
                    polygons.append([ 1,  4,  5,  8])
                    polygons.append([ 2,  7, 10, 11])
                    polygons.append([ 1,  8,  7,  2])
                else:
                    polygons.append([ 2,  1,  4, 11])
                    polygons.append([ 4,  5, 10, 11])
                    polygons.append([ 5,  8,  7, 10])
            else:
                polygons = [[ 1,  0,  3,  4],
                            [ 4,  5,  8,  7],
                            [ 7,  6,  9, 10],
                            [10, 11,  2,  1],
                            [ 1,  4,  7, 10]]

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


class DrillParser:
    class Tool:
        def __init__(self, number, diameter):
            self.number = number
            self.diameter = diameter

    def __init__(self):
        self.files = []
        self.tools = []
        self.holes = {}
        self.scale = 10.0

    #Add new file to drill file list
    def add(self, path):
        self.files.append(path)

    def readTools(self, stream):
        offset = len(self.tools) #FIXME Rewrite
        while True:
            data = stream.readline()
            eoh = re.search("^%$", data)
            if not len(data) or data[0] == "%":
                break
            tool = re.search("T(\d+)C([\.\d]+).*$", data, re.S)
            if tool and int(tool.group(1)) != 0:
                num, diam = int(tool.group(1)), (float(tool.group(2)) / 2) * self.scale
                self.tools.append(DrillParser.Tool(num + offset, diam))
        return offset

    def readSegments(self, stream, offset):
        current = None
        while True:
            data = stream.readline()
            if not len(data):
                break
            tool = re.search("T(\d+)$", data, re.S)
            if tool and int(tool.group(1)) != 0:
                current = None
                for item in self.tools: #TODO Rewrite
                    if item.number == int(tool.group(1)) + offset:
                        current = item
                        print current.number
                        self.holes[current.number] = []
                        break
            hole = re.search("X([\.\d]+)Y([\.\d]+)", data)
            if current is not None and hole:
                self.holes[current.number].append((float(hole.group(1)) * self.scale, \
                        float(hole.group(2)) * self.scale))

    def read(self):
        for path in self.files:
            stream = open(path, "rb")
            offset = self.readTools(stream)
            self.readSegments(stream, offset)
            stream.close()


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

def wrapTexture(mesh):
    bounds = [[mesh.vertices[0][0], mesh.vertices[0][1]], [mesh.vertices[0][0], mesh.vertices[0][1]]]
    for vert in mesh.vertices:
        if vert[0] < bounds[0][0]:
            bounds[0][0] = vert[0]
        if vert[0] > bounds[1][0]:
            bounds[1][0] = vert[0]
        if vert[1] < bounds[0][1]:
            bounds[0][1] = vert[1]
        if vert[1] > bounds[1][1]:
            bounds[1][1] = vert[1]
    print "Model boundaries: (%f, %f), (%f, %f)" % (bounds[0][0], bounds[0][1], bounds[1][0], bounds[1][1])
    size = (bounds[1][0] - bounds[0][0], bounds[1][1] - bounds[0][1])
    for poly in mesh.polygons:
        for index in poly:
            u = (mesh.vertices[index][0] - bounds[0][0]) / size[0]
            v = 1.0 - (mesh.vertices[index][1] - bounds[0][1]) / size[1]
            mesh.texels.append(numpy.array([u, v]))

def createBoard(vertices, polygons):
    top, bottom = model.Mesh(), model.Mesh()
    for vert in vertices:
        top.vertices.append(numpy.array([vert[0], vert[1], Rect.THICKNESS]))
        bottom.vertices.append(numpy.array([vert[0], vert[1], -Rect.THICKNESS]))
    for poly in polygons:
        top.polygons.append(poly[::])
        bottom.polygons.append(poly[::-1])
    return (top, bottom)

def createHole(radius):
    top, hole, bottom = model.Mesh(), model.Mesh(), model.Mesh()

    edges = 24
    angle, delta = 0, math.pi * 2 / edges
    for i in range(0, edges):
        xPos, yPos = radius * math.cos(angle), radius * math.sin(angle)
        top.vertices.append(numpy.array([xPos, yPos, Rect.THICKNESS]))
        bottom.vertices.append(numpy.array([xPos, yPos, -Rect.THICKNESS]))
        hole.vertices.extend([numpy.array([xPos, yPos, Rect.THICKNESS]), numpy.array([xPos, yPos, -Rect.THICKNESS])])
        angle += delta

    hole.polygons.append([(edges - 1) * 2, 0, 1, (edges - 1) * 2 + 1])
    for i in range(0, edges - 1):
        hole.polygons.append([(i + 0) * 2, (i + 1) * 2, (i + 1) * 2 + 1, (i + 0) * 2 + 1])

    planarCoords = [[radius, radius], [-radius, radius], [-radius, -radius], [radius, -radius]]
    for pair in planarCoords:
        top.vertices.append(numpy.array([pair[0], pair[1], Rect.THICKNESS]))
        bottom.vertices.append(numpy.array([pair[0], pair[1], -Rect.THICKNESS]))

    mult = edges / 4
    for i in range(0, mult):
        for j in range(0, 4):
            top.polygons.append([(i + 0) + mult * j, edges + j, (i + 1) + mult * j])
            bottom.polygons.append([(i + 1) + mult * j, edges + j, (i + 0) + mult * j])
    return (top, hole, bottom)

#def writeVRML(out, mesh, offset, img):
def writeVRML(out, mesh, offset, img = None):
    #print "Sizes xVert %u, xPoly %u, texVert %u, texPoly %u" % \
            #(len(vertices), len(polygons), len(texVertices), len(texPolygons))
    scale = (0.03937, 0.03937)
    out.write("#VRML V2.0 utf8\n#Created by b2m.py\n")
    out.write("DEF OB_%u Transform {\n" % random.randint(1000, 9999))
    out.write("\ttranslation 0 0 0\n")
    out.write("\trotation 1 0 0 0\n")
    out.write("\tscale 1 1 1\n")
    out.write("\tchildren [\n"
              "\t\tDEF ME_%u Group {\n"
              "\t\t\tchildren [\n"
              "\t\t\t\tShape {\n" % random.randint(1000, 9999))
    out.write("\t\t\t\t\tappearance Appearance {\n"
              "\t\t\t\t\t\tmaterial DEF MAT_%u Material {\n" % random.randint(1000, 9999))
    if img != None:
        out.write("\t\t\t\t\t\t\tdiffuseColor 1.0 1.0 1.0\n")
    else:
        out.write("\t\t\t\t\t\t\tdiffuseColor 0.039 0.138 0.332\n")
    out.write("\t\t\t\t\t\t\tambientIntensity 0.2\n"
              "\t\t\t\t\t\t\tspecularColor 1.0 1.0 1.0\n"
              "\t\t\t\t\t\t\temissiveColor  0.0 0.0 0.0\n"
              "\t\t\t\t\t\t\tshininess 0.5\n"
              "\t\t\t\t\t\t\ttransparency 0.0\n"
              "\t\t\t\t\t\t}\n");
    if img != None:
        out.write("\t\t\t\t\t\ttexture DEF diffusemap ImageTexture {\n"
                "\t\t\t\t\t\t\turl \"%s\"\n"
                "\t\t\t\t\t\t}\n" % img["diffuse"])
        out.write("\t\t\t\t\t\ttexture DEF normalmap ImageTexture {\n"
                "\t\t\t\t\t\t\turl \"%s\"\n"
                "\t\t\t\t\t\t}\n" % img["normals"])
    out.write("\t\t\t\t\t}\n")
    out.write("\t\t\t\t\tgeometry IndexedFaceSet {\n"
              "\t\t\t\t\t\tsolid FALSE\n"
              "\t\t\t\t\t\tcoord DEF coord_Cube Coordinate {\n"
              "\t\t\t\t\t\t\tpoint [\n")
    for v in mesh.vertices:
        out.write("%f %f %f\n" % ((v[0] + offset[0]) * scale[0], (v[1] + offset[1]) * scale[1], v[2]))
    out.write("\t\t\t\t\t\t\t]\n"
              "\t\t\t\t\t\t}\n"
              "\t\t\t\t\t\tcoordIndex [\n")
    for p in mesh.polygons:
        for index in p:
            out.write("%u " % index)
        out.write("-1,\n")

    if img != None:
        out.write("\t\t\t\t\t\t]\n");
        out.write("\t\t\t\t\t\ttexCoord TextureCoordinate {\n"
                "\t\t\t\t\t\tpoint [\n");
        for v in mesh.texels:
            out.write("%f %f,\n" % (v[0], v[1]))
        out.write("\t\t\t\t\t\t]\n");
        out.write("\t\t\t\t\t}\n");
        out.write("\t\t\t\t\ttexCoordIndex [\n");
        i = 0
        for p in mesh.polygons:
            for index in p:
                out.write("%u " % i)
                i += 1
            out.write("-1\n")
    out.write("\t\t\t\t\t\t]\n"
              "\t\t\t\t\t}\n"
              "\t\t\t\t}\n"
              "\t\t\t]\n"
              "\t\t}\n"
              "\t]\n"
              "}\n")

#random.seed()

parser = argparse.ArgumentParser()
parser.add_argument("-i", dest="path", help="project directory", default="")
parser.add_argument("-p", dest="project", help="project name", default="")
parser.add_argument("-o", dest="output", help="output directory", default="")
options = parser.parse_args()

if options.output == "":
    outPath = options.path
else:
    outPath = options.output

layerList = {}
for layer in [("Front", "F", "front"), ("Back", "B", "back")]:
    if os.path.isfile("%s%s-%s_Diffuse.png" % (outPath, options.project, layer[0])):
        layerList[layer[2]] = ({"diffuse": "%s%s-%s_Diffuse.png" % (outPath, options.project, layer[0]), \
                "normals": "%s%s-%s_Normals.png" % (outPath, options.project, layer[0])})

boardSize = (0, 0)
if layerList["front"] is not None: #FIXME Rewrite
    tmp = Image.open(layerList["front"]["diffuse"])
    #TODO Add variable DPI
    boardSize = (float(tmp.size[0]) / 900.0 * 254, float(tmp.size[1]) / 900.0 * 254)

boardCn = (boardSize[0] / 2, boardSize[1] / 2)

test = Rect(((0, 0), boardSize), ((0, 0), (0, 0), (0, 0), (0, 0)))
borders = model.Mesh()
borders.vertices, borders.polygons = test.borders()

dp = DrillParser()
for drillFile in ["", "-NPTH"]:
    filePath = "%s%s%s.drl" % (outPath, options.project, drillFile)
    if os.path.isfile(filePath):
        dp.add(filePath)
dp.read()

holeModels = {}
for tool in dp.tools:
    holeModels[tool.number] = createHole(tool.diameter)

for tool in dp.tools:
    for hole in dp.holes[tool.number]:
        test.subdivide(circleRect(hole, tool.diameter))

vert, poly = test.tesselate()
print "Complexity: vertices %u, polygons %u" % (len(vert), len(poly))
#vert, poly = optimizeVertices(vert, poly)

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

front, back = createBoard(vert, poly)
inner = model.Mesh()

for tool in dp.tools:
    for hole in dp.holes[tool.number]:
        holeTop = model.Mesh(holeModels[tool.number][0])
        holeCylinder = model.Mesh(holeModels[tool.number][1])
        holeBottom = model.Mesh(holeModels[tool.number][2])

        holeTop.translate((hole[0], hole[1], 0));
        holeCylinder.translate((hole[0], hole[1], 0));
        holeBottom.translate((hole[0], hole[1], 0));

        inner.append(holeCylinder)
        front.append(holeTop)
        back.append(holeBottom)

wrapTexture(front)
wrapTexture(back)
#wrapTexture(inner)
#wrapTexture(borders)

out = open("board.wrl", "wb")
#TODO Fix order
writeVRML(out, front, (-boardCn[0], -boardCn[1]), layerList["back"])
writeVRML(out, back, (-boardCn[0], -boardCn[1]), layerList["front"])
writeVRML(out, inner, (-boardCn[0], -boardCn[1]))
writeVRML(out, borders, (-boardCn[0], -boardCn[1]))
colorData.show()

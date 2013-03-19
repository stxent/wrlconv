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
import vrml_export
#import time
#import subprocess

THICKNESS = 0.1


class Rect:
    TOL = 1e-5 #Floating point values tolerance
    class RectCorner:
        def __init__(self, position, chamfer):
            self.position = numpy.array([position[0], position[1]])
            self.chamfer = numpy.array([chamfer[0], chamfer[1]])

        def generate(self, vect):
            vertices = []
            vertices.append(numpy.array([self.position[0] + self.chamfer[0] * vect[0], self.position[1], 0]))
            vertices.append(numpy.array([self.position[0] + self.chamfer[0] * vect[0],
                    self.position[1] + self.chamfer[1] * vect[1], 0]))
            vertices.append(numpy.array([self.position[0], self.position[1] + self.chamfer[1] * vect[1], 0]))
            return vertices

    #Check intersection of rectanle and point
    @staticmethod
    def prCollision(rect, point):
        top, bottom = rect[0], rect[1]
        return top[0] <= point[0] <= bottom[0] and top[1] <= point[1] <= bottom[1]

    #Check intersection of two rectangles
    @staticmethod
    def rCollision(ra, rb):
        return (ra[0][0] <= rb[0][0] <= ra[1][0] or rb[0][0] <= ra[0][0] <= rb[1][0]) and \
               (ra[0][1] <= rb[0][1] <= ra[1][1] or rb[0][1] <= ra[0][1] <= rb[1][1])

    def __init__(self, points, chamfers):
        self.coords = points #Coordinates of top left and bottom right corners
        self.corners = []
        self.sub = None
        for i in range(0, 4):
            self.corners.append(Rect.RectCorner(numpy.array([0, 0]), numpy.array([chamfers[i][0], chamfers[i][1]])))
        self.recalcCorners()

    def recalcCorners(self):
        cpoints = [[self.coords[0][0], self.coords[0][1]],
                   [self.coords[1][0], self.coords[0][1]],
                   [self.coords[1][0], self.coords[1][1]],
                   [self.coords[0][0], self.coords[1][1]]]
        for i in range(0, len(self.corners)):
            self.corners[i].position = numpy.array(cpoints[i])

    def contain(self, points):
        return Rect.prCollision(self.coords, points[0]) and Rect.prCollision(self.coords, points[1])

    def intersectEdge(self, points):
        #Returns intersected edge: 0 top, 1 right, 2 bottom, 3 left
        #Horizontal edges
        if self.coords[0][0] <= points[0][0] and points[1][0] <= self.coords[1][0]:
            if points[0][1] <= self.coords[0][1] <= points[1][1]:
                return 0 #Top
            if points[0][1] <= self.coords[1][1] <= points[1][1]:
                return 2 #Bottom
        #Vertical edges
        if self.coords[0][1] <= points[0][1] and points[1][1] <= self.coords[1][1]:
            if points[0][0] <= self.coords[0][0] <= points[1][0]:
                return 3 #Left
            if points[0][0] <= self.coords[1][0] <= points[1][0]:
                return 1 #Right
        return None

    def intersectCorner(self, points):
        if Rect.prCollision(points, self.coords[0]):
            return 0 #Top left
        if Rect.prCollision(points, (self.coords[1][0], self.coords[0][1])):
            return 1 #Top right
        if Rect.prCollision(points, self.coords[1]):
            return 2 #Bottom left
        if Rect.prCollision(points, (self.coords[0][0], self.coords[1][1])):
            return 3 #Bottom right
        return None

    def tesselate(self):
        #Returns tuple with vertex and polygon lists
        if self.sub is None:
            vertices, polygons = [], []

            #Left edges
            inList = []
            offsetTop, offsetBottom = 0, 0
            if self.corners[0].chamfer[0] > Rect.TOL and self.corners[0].chamfer[1] > Rect.TOL:
                inList.append((self.coords[0] + self.corners[0].chamfer * numpy.array([1, 0]), \
                        self.coords[0] + self.corners[0].chamfer))
                offsetTop = self.corners[0].chamfer[1]
            if self.corners[3].chamfer[0] > Rect.TOL and self.corners[3].chamfer[1] > Rect.TOL:
                inList.append((numpy.array([self.coords[0][0] + self.corners[3].chamfer[0], \
                        self.coords[1][1] - self.corners[3].chamfer[1]]), \
                        numpy.array([self.coords[0][0] + self.corners[3].chamfer[0], self.coords[1][1]])))
                offsetBottom = self.corners[3].chamfer[1]
            inList.append((numpy.array([self.coords[0][0], self.coords[0][1] + offsetTop]), \
                    numpy.array([self.coords[0][0], self.coords[1][1] - offsetBottom])))

            #Right edges
            outList = []
            offsetTop, offsetBottom = 0, 0
            if self.corners[1].chamfer[0] > Rect.TOL and self.corners[1].chamfer[1] > Rect.TOL:
                outList.append((numpy.array([self.coords[1][0] - self.corners[1].chamfer[0], \
                        self.coords[0][1]]), numpy.array([self.coords[1][0] - self.corners[1].chamfer[0], \
                        self.coords[0][1] + self.corners[1].chamfer[1]])))
                offsetTop = self.corners[1].chamfer[1]
            if self.corners[2].chamfer[0] > Rect.TOL and self.corners[2].chamfer[1] > Rect.TOL:
                outList.append((self.coords[1] - self.corners[2].chamfer, \
                        self.coords[1] + self.corners[2].chamfer * numpy.array([-1, 0])))
                offsetBottom = self.corners[2].chamfer[1]
            outList.append((numpy.array([self.coords[1][0], self.coords[0][1] + offsetTop]), \
                    numpy.array([self.coords[1][0], self.coords[1][1] - offsetBottom])))

            #Possible intersections
            hPoints = []
            hPoints.append(self.coords[1][0])
            if math.fabs(self.corners[1].chamfer[0] - self.corners[2].chamfer[0]) < Rect.TOL:
                if self.corners[1].chamfer[0] > Rect.TOL:
                    hPoints.append(self.coords[1][0] - self.corners[1].chamfer[0])
            else:
                if self.corners[1].chamfer[0] > Rect.TOL:
                    hPoints.append(self.coords[1][0] - self.corners[1].chamfer[0])
                if self.corners[2].chamfer[0] > Rect.TOL:
                    hPoints.append(self.coords[1][0] - self.corners[2].chamfer[0])
            hPoints = sorted(hPoints)

            #Check intersection of line and point, only for vertical lines
            def lpCollision(line, point):
                if line[0][0] == point[0] and line[0][1] <= point[1] <= line[1][1]:
                    return True
                return False

            while True:
                edgeDivided = False
                i = 0
                while i < len(inList):
                    for gap in outList:
                        edge = inList[i]
                        mEdge = (numpy.array([gap[0][0], edge[0][1]]), numpy.array([gap[1][0], edge[1][1]]))
                        if lpCollision(gap, mEdge[0]) or lpCollision(gap, mEdge[1]):
                            if min(edge[1][1], gap[1][1]) - max(edge[0][1], gap[0][1]) < Rect.TOL:
                                continue
                            inList.pop(i)
                            edgeDivided = True
                            start, end = None, None

                            offset = gap[0][1] - edge[0][1]
                            if offset > Rect.TOL:
                                inList.append((numpy.array([edge[0][0], edge[0][1]]), \
                                        numpy.array([edge[0][0], edge[0][1] + offset])))
                                start = numpy.array([gap[0][0], edge[0][1] + offset])
                            else:
                                start = numpy.array([gap[0][0], edge[0][1]])

                            offset = edge[1][1] - gap[1][1]
                            if offset > Rect.TOL:
                                inList.append((numpy.array([edge[1][0], edge[1][1] - offset]), \
                                        numpy.array([edge[1][0], edge[1][1]])))
                                end = numpy.array([gap[1][0], edge[1][1] - offset])
                            else:
                                end = numpy.array([gap[1][0], edge[1][1]])

                            if not numpy.allclose(end - start, 0.):
                                vList = range(len(vertices), len(vertices) + 4)
                                vertices.append(numpy.array([edge[0][0], start[1], 0]))
                                vertices.append(numpy.array([start[0], start[1], 0]))
                                vertices.append(numpy.array([end[0], end[1], 0]))
                                vertices.append(numpy.array([edge[1][0], end[1], 0]))
                                polygons.append([vList[0], vList[1], vList[2], vList[3]])
                            break
                    if edgeDivided:
                        break
                    i += 1
                if i == len(inList):
                    break

            rebuilded = []
            vIndex = range(0, len(vertices))
            while len(vIndex):
                vert = vertices[vIndex[0]]
                same = []
                for i in range(0, len(vertices)):
                    if numpy.allclose(vertices[i], vert):
                        same.append(i)
                last = len(rebuilded)
                for poly in polygons:
                    for i in range(0, len(poly)):
                        if poly[i] in same:
                            poly[i] = last
                for ind in same:
                    vIndex.remove(ind)
                rebuilded.append(vert)
            return (rebuilded, polygons)
        else:
            vertices, polygons = [], []
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
            count = len(vertices)
            vertices.extend(self.corners[0].generate(( 1,  1)))
            vertices.extend(self.corners[1].generate((-1,  1)))
            vertices.extend(self.corners[2].generate((-1, -1)))
            vertices.extend(self.corners[3].generate(( 1, -1)))
            for i in range(0, count):
                vertices[i][2] = -THICKNESS
            for i in range(count, count * 2):
                vertices[i][2] = THICKNESS
            return (vertices, polygons)
        else:
            return ([], [])

    def simplify(self):
        size = (self.coords[1][0] - self.coords[0][0], self.coords[1][1] - self.coords[0][1])
        for i in range(0, len(self.corners)):
            if math.fabs(self.corners[i].chamfer[1] - size[1]) < Rect.TOL:
                if i in (0, 3):
                    self.coords = ((self.coords[0][0] + self.corners[i].chamfer[0], self.coords[0][1]), self.coords[1])
                    self.corners[0].chamfer, self.corners[3].chamfer = numpy.array([0, 0]), numpy.array([0, 0])
                if i in (1, 2):
                    self.coords = (self.coords[0], (self.coords[1][0] - self.corners[i].chamfer[0], self.coords[1][1]))
                    self.corners[1].chamfer, self.corners[2].chamfer = numpy.array([0, 0]), numpy.array([0, 0])
            if math.fabs(self.corners[i].chamfer[0] - size[0]) < Rect.TOL:
                if i in (0, 1):
                    self.coords = ((self.coords[0][0], self.coords[0][1] + self.corners[i].chamfer[1]), self.coords[1])
                    self.corners[0].chamfer, self.corners[1].chamfer = numpy.array([0, 0]), numpy.array([0, 0])
                if i in (2, 3):
                    self.coords = (self.coords[0], (self.coords[1][0], self.coords[1][1] - self.corners[i].chamfer[1]))
                    self.corners[2].chamfer, self.corners[3].chamfer = numpy.array([0, 0]), numpy.array([0, 0])
        self.recalcCorners()

    def subdivide(self, points):
        size = (points[1][0] - points[0][0], points[1][1] - points[0][1])
        center = (points[0][0] + size[0] / 2, points[0][1] + size[1] / 2)
        if self.sub is None:
            #TODO Add case when the hole contains full rectangle
            inner = ((self.coords[0][0] + max(self.corners[0].chamfer[0], self.corners[3].chamfer[0]), \
                    self.coords[0][1] + max(self.corners[0].chamfer[1], self.corners[1].chamfer[1])),
                    (self.coords[1][0] - max(self.corners[1].chamfer[0], self.corners[2].chamfer[0]), \
                    self.coords[1][1] - max(self.corners[2].chamfer[1], self.corners[3].chamfer[1])))
            if inner[0][0] > inner[1][0] or inner[0][1] > inner[1][1]:
                inner = None
            if inner is not None and Rect.prCollision(self.coords, center) and not Rect.prCollision(inner, center):
                #Dead zone
                sideRects = [(self.coords[0], (inner[0][0], self.coords[1][1])), \
                        ((inner[1][0], self.coords[0][1]), self.coords[1]), \
                        (self.coords[0], (self.coords[1][0], inner[0][1])), \
                        ((self.coords[0][0], inner[1][1]), self.coords[1])]

                edge = None
                for i in range(0, len(sideRects)):
                    if Rect.prCollision(sideRects[i], center):
                        edge = i
                        break

                pim = (0, 1) if edge in (1, 3) else (1, 0) #Imaginary part
                pre = (0, 1) if edge in (2, 0) else (1, 0) #Real part

                coords, chamfers = [], []
                if edge == 0:
                    val = self.coords[0][1] + max(self.corners[0].chamfer[1], self.corners[1].chamfer[1])
                if edge == 1:
                    val = self.coords[1][0] - max(self.corners[1].chamfer[0], self.corners[2].chamfer[0])
                if edge == 2:
                    val = self.coords[1][1] - max(self.corners[2].chamfer[1], self.corners[3].chamfer[1])
                if edge == 3:
                    val = self.coords[0][0] + max(self.corners[3].chamfer[0], self.corners[0].chamfer[0])

                pts = (self.coords[0][pim[0]], val, self.coords[1][pim[0]])
                if pts[1] - pts[0] > Rect.TOL and pts[2] - pts[1] > Rect.TOL:
                    self.sub = []
                    if pim[0] == 1:
                        chamfers.append((self.corners[0].chamfer, self.corners[1].chamfer, (0, 0), (0, 0)))
                        chamfers.append(((0, 0), (0, 0), self.corners[2].chamfer, self.corners[3].chamfer))
                        coords.append(((self.coords[0][0], pts[0]), (self.coords[1][0], pts[1])))
                        coords.append(((self.coords[0][0], pts[1]), (self.coords[1][0], pts[2])))
                    else:
                        chamfers.append((self.corners[0].chamfer, (0, 0), (0, 0), self.corners[3].chamfer))
                        chamfers.append(((0, 0), self.corners[1].chamfer, self.corners[2].chamfer, (0, 0)))
                        coords.append(((pts[0], self.coords[0][1]), (pts[1], self.coords[1][1])))
                        coords.append(((pts[1], self.coords[0][1]), (pts[2], self.coords[1][1])))
                    self.sub.extend([Rect(coords[0], chamfers[0]), Rect(coords[1], chamfers[1])])

                    for entry in self.sub:
                        entry.simplify()
                        entry.subdivide(points)
                    return

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
                for entry in self.sub:
                    entry.simplify()
                return

            corner = self.intersectCorner(points)
            if corner is not None:
                index = ((1, 0, 1, 0), (0, 1, 1, 0), (0, 1, 0, 1), (1, 0, 0, 1))[corner]
                dx = max(self.corners[corner].chamfer[0], math.fabs(points[index[0]][0] - self.coords[index[1]][0]))
                dx = min(dx, self.coords[1][0] - self.coords[0][0])
                dy = max(self.corners[corner].chamfer[1], math.fabs(points[index[2]][1] - self.coords[index[3]][1]))
                dy = min(dy, self.coords[1][1] - self.coords[0][1])
                self.corners[corner].chamfer = numpy.array([dx, dy])
                self.simplify()
                return

            edge = self.intersectEdge(points)
            if edge is not None:
                self.sub = []
                pim = (0, 1) if edge in (0, 2) else (1, 0) #Imaginary part
                pre = (0, 1) if edge in (1, 2) else (1, 0) #Real part
                ach = size[pim[0]] / 2
                bch = math.fabs(self.coords[pre[1]][pim[1]] - points[pre[0]][pim[1]])
                bch = min(bch, self.coords[1][pim[1]] - self.coords[0][pim[1]])
                if pim[0] == 1:
                    ach, bch = bch, ach
                value = {pre[1]: (ach, bch), pre[0]: (0, 0)}

                coords, chamfers = [], []
                varPart = ((center[pim[0]], self.coords[1][pim[1]]),
                        (center[pim[0]], self.coords[0][pim[1]]))
                coords.append((self.coords[0], (varPart[0][pim[0]], varPart[0][pim[1]])))
                coords.append(((varPart[1][pim[0]], varPart[1][pim[1]]), self.coords[1]))

                if pim[0] == 0:
                    chamfers.append((self.corners[0].chamfer, value[0], value[1], self.corners[3].chamfer))
                    chamfers.append((value[0], self.corners[1].chamfer, self.corners[2].chamfer, value[1]))
                else:
                    chamfers.append((self.corners[0].chamfer, self.corners[1].chamfer, value[1], value[0]))
                    chamfers.append((value[0], value[1], self.corners[2].chamfer, self.corners[3].chamfer))
                self.sub.extend([Rect(coords[0], chamfers[0]), Rect(coords[1], chamfers[1])])
                for entry in self.sub:
                    entry.simplify()
                return
        else:
            if Rect.rCollision(self.coords, points):
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
        self.scale = 10.0 #FIXME

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
                        self.holes[current.number] = []
                        break
            hole = re.search("X([\.\d]+)Y([\.\d]+)", data)
            if current is not None and hole:
                self.holes[current.number].append(numpy.array([float(hole.group(1)) * self.scale, \
                        float(hole.group(2)) * self.scale]))

    def read(self):
        for path in self.files:
            stream = open(path, "rb")
            offset = self.readTools(stream)
            self.readSegments(stream, offset)
            stream.close()


#FIXME Move to separate module
def calcBorders(path):
    stream = open(path, "rb")
    edgeCuts = stream.read()
    stream.close()
    lineSearch = re.compile("<path.+?d=\"(.*?)\".*?>", re.S)
    pos = 0
    position = [None, None]
    while 1:
        content = lineSearch.search(edgeCuts, pos)
        if content is None:
            break
        data = []
        unformatted = content.group(1).replace("M", "").replace("L", "").splitlines()
        for part in unformatted:
            data.extend(part.split(' '))
        coords = ((float(data[0]), float(data[1])), (float(data[2]), float(data[3])))
        for point in coords:
            if position[0] is None:
                position[0] = numpy.array([point[0], point[1]])
            else:
                if position[0][0] > point[0]:
                    position[0][0] = point[0]
                if position[0][1] > point[1]:
                    position[0][1] = point[1]
            if position[1] is None:
                position[1] = numpy.array([point[0], point[1]])
            else:
                if position[1][0] < point[0]:
                    position[1][0] = point[0]
                if position[1][1] < point[1]:
                    position[1][1] = point[1]
        pos = content.end()
    return position

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
    top, bottom = model.Mesh(), model.Mesh()
    for vert in vertices:
        top.vertices.append(numpy.array([vert[0], vert[1], THICKNESS]))
        bottom.vertices.append(numpy.array([vert[0], vert[1], -THICKNESS]))
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
        top.vertices.append(numpy.array([xPos, yPos, THICKNESS]))
        bottom.vertices.append(numpy.array([xPos, yPos, -THICKNESS]))
        hole.vertices.extend([numpy.array([xPos, yPos, THICKNESS]), numpy.array([xPos, yPos, -THICKNESS])])
        angle += delta

    hole.polygons.append([(edges - 1) * 2, 0, 1, (edges - 1) * 2 + 1])
    for i in range(0, edges - 1):
        hole.polygons.append([(i + 0) * 2, (i + 1) * 2, (i + 1) * 2 + 1, (i + 0) * 2 + 1])

    planarCoords = [[radius, radius], [-radius, radius], [-radius, -radius], [radius, -radius]]
    for pair in planarCoords:
        top.vertices.append(numpy.array([pair[0], pair[1], THICKNESS]))
        bottom.vertices.append(numpy.array([pair[0], pair[1], -THICKNESS]))

    mult = edges / 4
    for i in range(0, mult):
        for j in range(0, 4):
            top.polygons.append([(i + 0) + mult * j, edges + j, (i + 1) + mult * j])
            bottom.polygons.append([(i + 1) + mult * j, edges + j, (i + 0) + mult * j])
    return (top, hole, bottom)

parser = argparse.ArgumentParser()
parser.add_argument("-i", dest="path", help="project directory", default="")
parser.add_argument("-p", dest="project", help="project name", default="")
parser.add_argument("-o", dest="output", help="output directory", default="")
parser.add_argument("--mask", dest="mask", help="mask color", default="10,35,85")
parser.add_argument("--silk", dest="silk", help="silk color", default="255,255,255")
parser.add_argument("--plating", dest="plating", help="plating color", default="255,228,0")
options = parser.parse_args()

if options.output == "":
    outPath = options.path
else:
    outPath = options.output

colors = {"mask": (), "silk": (), "plating": ()}
for color in [("mask", options.mask), ("silk", options.silk), ("plating", options.plating)]:
    splitted = color[1].split(",")
    #TODO Add value checking
    try:
        colors[color[0]] = numpy.array([float(splitted[0]) / 256, float(splitted[1]) / 256, float(splitted[2]) / 256])
    except ValueError:
        print "Wrong color parameter: %s" % color[1]

layerList = {}
for layer in [("Front", "F", "front"), ("Back", "B", "back")]:
    layerFile = "%s%s-%s_Diffuse.png" % (outPath, options.project, layer[0])
    if os.path.isfile(layerFile):
        layerList[layer[2]] = ({"diffuse": "%s-%s_Diffuse.png" % (options.project, layer[0]), \
                "normalmap": "%s-%s_Normals.png" % (options.project, layer[0])})
    else:
        print "Layer file does not exist: %s" % layerFile

boardSize = (0, 0)
edgeFile = "%s%s-Edge_Cuts.svg" % (options.path, options.project)
if os.path.isfile(edgeFile):
    boardPos = calcBorders(edgeFile)
    boardSize = (boardPos[1] - boardPos[0]) / 1000 * 2.54 #FIXME Scale
else:
    print "% not found" % edgeFile
    exit()

if len(layerList) > 0:
    layer = layerList.itervalues().next()
    tmp = Image.open(outPath + layer["diffuse"])
    #TODO Add variable DPI
    boardSize = (float(tmp.size[0]) / 900.0 * 254, float(tmp.size[1]) / 900.0 * 254)
else:
    print "No copper layers found"
    exit()

boardCn = numpy.array([boardSize[0] / 2, boardSize[1] / 2])

test = Rect((-boardCn, +boardCn), ((0, 0), (0, 0), (0, 0), (0, 0)))
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
        test.subdivide(circleRect((hole[0] - boardCn[0], -(hole[1] - boardCn[1])), tool.diameter))

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
inner.smooth = True

for tool in dp.tools:
    for hole in dp.holes[tool.number]:
        holeTop = model.Mesh(holeModels[tool.number][0])
        holeCylinder = model.Mesh(holeModels[tool.number][1])
        holeBottom = model.Mesh(holeModels[tool.number][2])

        holeTop.translate((hole[0] - boardCn[0], -(hole[1] - boardCn[1]), 0));
        holeCylinder.translate((hole[0] - boardCn[0], -(hole[1] - boardCn[1]), 0));
        holeBottom.translate((hole[0] - boardCn[0], -(hole[1] - boardCn[1]), 0));

        inner.append(holeCylinder)
        front.append(holeTop)
        back.append(holeBottom)

model.uvWrapPlanar(front)
model.uvWrapPlanar(back)
model.uvWrapPlanar(inner, (-boardCn, +boardCn))

boardColor = model.Material.Color()
boardColor.diffuse = colors["mask"]
boardColor.ambient = colors["mask"] / 5
boardColor.shininess = 0.95
boardColor.specular = boardColor.diffuse
borders.material.color = boardColor

texColor = model.Material.Color()
texColor.shininess = 0.95
texColor.specular = texColor.diffuse
front.material.color = texColor
back.material.color = texColor
inner.material.color = texColor

#TODO Fix order
if "front" in layerList:
    front.material.diffuse = model.Material.Texture(layerList["front"]["diffuse"])
    front.material.normalmap = model.Material.Texture(layerList["front"]["normalmap"])
else:
    front.material.color = boardColor
if "back" in layerList:
    back.material.diffuse = model.Material.Texture(layerList["back"]["diffuse"])
    back.material.normalmap = model.Material.Texture(layerList["back"]["normalmap"])
else:
    back.material.color = boardColor

if "front" in layerList:
    inner.material.diffuse = model.Material.Texture(layerList["front"]["diffuse"])
elif "back" in layerList:
    inner.material.diffuse = model.Material.Texture(layerList["back"]["diffuse"])
else:
    inner.material.color = boardColor

exportList = [front, back, inner, borders]
for entry in exportList:
    entry.scale((0.025, 0.025, 1.0))

vrml_export.exportVRML("%s%s.wrl" % (outPath, options.project), exportList)

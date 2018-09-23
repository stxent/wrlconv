#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# model.py
# Copyright (C) 2013 xent
# Project is distributed under the terms of the GNU General Public License v3.0

import copy
import math
import numpy

def normalize(v):
    length = numpy.linalg.norm(v)
    return v if length == 0.0 else v / length

def createModelViewMatrix(eye, center, up):
    eye = numpy.array(eye)[:,0][0:3]
    center = numpy.array(center)[:,0][0:3]
    up = numpy.array(up)[:,0][0:3]

    forward = normalize(center - eye)
    up = normalize(up)
    side = numpy.cross(forward, up)
    side = numpy.array([0.0, 1.0, 0.0]) if numpy.linalg.norm(side) == 0.0 else normalize(side)
    up = numpy.cross(side, forward)
    up = normalize(up)

    result = numpy.matrix([
            [    1.0,     0.0,     0.0, 0.0],
            [    0.0,     1.0,     0.0, 0.0],
            [    0.0,     0.0,     1.0, 0.0],
            [-eye[0], -eye[1], -eye[2], 1.0]])
    result *= numpy.matrix([
            [side[0], up[0], -forward[0], 0.0],
            [side[1], up[1], -forward[1], 0.0],
            [side[2], up[2], -forward[2], 0.0],
            [    0.0,   0.0,         0.0, 1.0]])
    return result

def createPerspectiveMatrix(aspect, angle, distance):
    n, f = distance
    fov = angle * math.pi / 720.0
    h = 1.0 / math.tan(fov)
    w = h / aspect
    return numpy.matrix([
            [  w, 0.0,                      0.0,  0.0],
            [0.0,   h,                      0.0,  0.0],
            [0.0, 0.0,       -(f + n) / (f - n), -1.0],
            [0.0, 0.0, -(2.0 * f * n) / (f - n),  0.0]])

def createOrthographicMatrix(area, distance):
    n, f = distance
    w, h = area
    return numpy.matrix([
            [1.0 / w,     0.0,                0.0, 0.0],
            [    0.0, 1.0 / h,                0.0, 0.0],
            [    0.0,     0.0,     -2.0 / (f - n), 0.0],
            [    0.0,     0.0, -(f + n) / (f - n), 1.0]])

def metricToImperial(values):
    # Convert millimeters to hundreds of mils
    try:
        return [v / 2.54 for v in values]
    except TypeError:
        return values / 2.54

def uvWrapPlanar(mesh, borders=None):
    if borders is None:
        borders = [[mesh.geoVertices[0][0], mesh.geoVertices[0][1]], [mesh.geoVertices[0][0], mesh.geoVertices[0][1]]]
        for vert in mesh.geoVertices:
            if vert[0] < borders[0][0]:
                borders[0][0] = vert[0]
            if vert[0] > borders[1][0]:
                borders[1][0] = vert[0]
            if vert[1] < borders[0][1]:
                borders[0][1] = vert[1]
            if vert[1] > borders[1][1]:
                borders[1][1] = vert[1]
    size = (borders[1][0] - borders[0][0], borders[1][1] - borders[0][1])
    for poly in mesh.geoPolygons:
        for index in poly:
            u = (mesh.geoVertices[index][0] - borders[0][0]) / size[0]
            v = (mesh.geoVertices[index][1] - borders[0][1]) / size[1]
            mesh.texVertices.append(numpy.array([u, v]))
        mesh.texPolygons.append(poly)

def angle(v1, v2):
    mag1 = math.sqrt(v1[0] * v1[0] + v1[1] * v1[1] + v1[2] * v1[2])
    mag2 = math.sqrt(v2[0] * v2[0] + v2[1] * v2[1] + v2[2] * v2[2])
    mag = mag1 * mag2
    if mag == 0.0:
        return 0.0
    res = (v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2]) / mag
    ac = math.acos(res)
    if v2[0] * v1[1] - v2[1] * v1[0] < 0.0:
        ac *= -1.0
    return ac

def divideTriangleByPlane(triangle, p, n):
    def squashPatch(patch):
        output = []
        for i in range(0, len(patch)):
            unique = True
            for j in range(0, len(output)):
                if Mesh.comparePoints(patch[i], output[j]):
                    unique = False
            if unique:
                output.append(patch[i])
        return output

    crosses = [
        intersectLinePlane(p, n, triangle[1], triangle[2]),
        intersectLinePlane(p, n, triangle[2], triangle[0]),
        intersectLinePlane(p, n, triangle[0], triangle[1])
    ]
    raw = [i for i in range(0, len(crosses)) if crosses[i] is None]

    if len(raw) == 3:
        return [triangle]
    elif len(raw) == 2:
        crossed = [i for i in range(0, 3) if i not in raw]
        tri0 = squashPatch([triangle[crossed[0]], triangle[raw[0]], crosses[crossed[0]]])
        tri1 = squashPatch([triangle[raw[1]], triangle[crossed[0]], crosses[crossed[0]]])
        return filter(lambda x: len(x) > 2, [tri0, tri1])
    elif len(raw) == 1:
        crossed = [raw[0] + 1 if raw[0] < 2 else 0, raw[0] - 1 if raw[0] > 0 else 2]
        tri = squashPatch([triangle[raw[0]], crosses[crossed[1]], crosses[crossed[0]]])
        quad = squashPatch([triangle[crossed[0]], triangle[crossed[1]], crosses[crossed[0]], crosses[crossed[1]]])
        return filter(lambda x: len(x) > 2, [tri, quad])
    else:
        return []

def divideByPlane(patch, p, n):
    if len(patch) == 3:
        return divideTriangleByPlane(patch, p, n)
    else:
        triangles = []
        triangles.extend(map(lambda x: [patch[0], patch[x], patch[x + 1]], range(1, len(patch) - 1)))

        output = []
        [output.extend(divideTriangleByPlane(tri, p, n)) for tri in triangles]

        if len(triangles) == len(output):
            unique = False
            for i in range(0, len(triangles)):
                if len(triangles[i]) == len(output[i]):
                    for j in range(0, len(triangles[i])):
                        if not Mesh.comparePoints(triangles[i][j], output[i][j]):
                            unique = True
                            break
                    if unique:
                        break
                else:
                    unique = True
                    break
            if not unique:
                return [patch]

        return output

def intersectLinePlane(planePoint, planeNormal, lineStart, lineEnd):
    lineVector = normalize(lineEnd - lineStart)
    if numpy.dot(planeNormal, lineVector) == 0.0:
        return None
    lineLength = numpy.linalg.norm(lineEnd - lineStart)
    t = numpy.dot(planeNormal, planePoint - lineStart) / numpy.dot(planeNormal, lineVector)
    if t <= 0.0 or t >= lineLength:
        return None
    else:
        return lineVector * t + lineStart

def normal(v1, v2):
    return numpy.cross(v1[0:3], v2[0:3])

def tangent(v1, v2, st1, st2):
    div = st1[1] * st2[0] - st1[0] * st2[1]
    if div != 0:
        coef = 1.0 / div
        return (v1[0:3] * -st2[1] + v2[0:3] * st1[1]) * coef
    else:
        return numpy.array([0.0, 0.0, 1.0])

def rotationMatrix(v, angle):
    cs, sn = math.cos(angle), math.sin(angle)
    v = numpy.asfarray(v)

    a11 = cs + v[0] * v[0] * (1.0 - cs)
    a12 = v[0] * v[1] * (1.0 - cs) - v[2] * sn
    a13 = v[0] * v[2] * (1.0 - cs) + v[1] * sn
    a21 = v[1] * v[0] * (1.0 - cs) + v[2] * sn
    a22 = cs + v[1] * v[1] * (1.0 - cs)
    a23 = v[1] * v[2] * (1.0 - cs) - v[0] * sn
    a31 = v[2] * v[0] * (1.0 - cs) - v[1] * sn
    a32 = v[2] * v[1] * (1.0 - cs) + v[0] * sn
    a33 = cs + v[2] * v[2] * (1.0 - cs)

    return numpy.matrix([
            [a11, a12, a13, 0.0],
            [a21, a22, a23, 0.0],
            [a31, a32, a33, 0.0],
            [0.0, 0.0, 0.0, 1.0]])


class Material:
    class Color:
        IDENT = 0
        TOLERANCE = 0.001

        def __init__(self, name=None):
            self.diffuse = numpy.array([1.0, 1.0, 1.0])
            self.ambient = numpy.array([0.0, 0.0, 0.0])
            self.specular = numpy.array([0.0, 0.0, 0.0])
            self.emissive = numpy.array([0.0, 0.0, 0.0])
            self.shininess = 0.0
            self.transparency = 0.0
            if name is None:
                self.ident = str(Material.Color.IDENT)
                Material.Color.IDENT += 1
            else:
                self.ident = name

        def __eq__(self, other):
            if not isinstance(other, Material.Color):
                return False
            def eq(a, b):
                return a - Material.Color.TOLERANCE <= b <= a + Material.Color.TOLERANCE
            def eqv(a, b):
                return eq(a[0], b[0]) and eq(a[1], b[1]) and eq(a[2], b[2])
            return (eq(self.transparency, other.transparency)
                    and eqv(self.diffuse, other.diffuse)
                    and eqv(self.ambient, other.ambient)
                    and eqv(self.specular, other.specular)
                    and eqv(self.emissive, other.emissive)
                    and eq(self.shininess, other.shininess))

        def __ne__(self, other):
            return not self == other


    class Texture:
        IDENT = 0

        def __init__(self, path, name=None):
            self.path = path
            if name is None:
                self.ident = str(Material.Texture.IDENT)
                Material.Texture.IDENT += 1
            else:
                self.ident = name

        def __eq__(self, other):
            if not isinstance(other, Material.Texture):
                return False
            return self.path == other.path

        def __ne__(self, other):
            return not self == other


    def __init__(self):
        self.color = Material.Color()
        self.diffuse = None
        self.normal = None
        self.specular = None

    def __eq__(self, other):
        if isinstance(other, Material):
            return self.color == other.color\
                    and self.diffuse == other.diffuse\
                    and self.normal == other.normal\
                    and self.specular == other.specular
        else:
            return False

    def __ne__(self, other):
        return not self == other


class Object:
    POINTS, LINES, PATCHES = range(0, 3)
    IDENT = 0

    def __init__(self, style, parent=None, name=None):
        self.transform = None
        self.parent = parent
        self.style = style

        if name is None:
            self.ident = str(Object.IDENT)
            Object.IDENT += 1
        else:
            self.ident = name

    def translate(self, arg):
        if self.transform is None:
            self.transform = Transform()
        self.transform.translate(arg)

    def rotate(self, vector, angle):
        if self.transform is None:
            self.transform = Transform()
        self.transform.rotate(vector, angle)

    def scale(self, arg):
        if self.transform is None:
            self.transform = Transform()
        self.transform.scale(arg)


class Mesh(Object):
    class Appearance:
        def __init__(self):
            self.material = Material()
            self.normals = False

            self.smooth = False
            self.solid = False
            self.wireframe = False


    def __init__(self, parent=None, name=None):
        Object.__init__(self, Object.PATCHES, parent, name)

        if self.parent is None:
            self.geoVertices, self.geoPolygons = [], []
            self.texVertices, self.texPolygons = [], []
            self.visualAppearance = Mesh.Appearance()

    def appearance(self):
        return self.visualAppearance if self.parent is None else self.parent.appearance()

    def geometry(self):
        return (self.geoVertices, self.geoPolygons) if self.parent is None else self.parent.geometry()

    def texture(self):
        return (self.texVertices, self.texPolygons) if self.parent is None else self.parent.texture()

    def isTextured(self):
        if self.parent is None:
            return len(self.texPolygons) > 0 and len(self.geoPolygons) == len(self.texPolygons)
        else:
            return self.parent.isTextured()

    def append(self, other):
        geoSize = len(self.geoVertices)
        geoVertices, geoPolygons = other.geometry()

        for entry in geoPolygons:
            self.geoPolygons.append([geoSize + vertex for vertex in entry])
        if other.transform is None:
            self.geoVertices += geoVertices
        else:
            self.geoVertices += [other.transform.apply(v) for v in geoVertices]

        texSize = len(self.texVertices)
        texVertices, texPolygons = other.texture()

        for entry in texPolygons:
            self.texPolygons.append([texSize + vertex for vertex in entry])
        self.texVertices += texVertices

    def applyTransform(self, transform=None):
        if transform is None:
            transform = self.transform
            self.transform = None
        if transform is not None:
            self.geoVertices = [transform.apply(v) for v in self.geoVertices]

    def optimize(self):
        if self.parent is not None:
            return

        #TODO Reduce complexity
        retVert = []
        retPoly = copy.deepcopy(self.geoPolygons)
        vIndex = list(range(0, len(self.geoVertices)))
        while len(vIndex):
            vert = self.geoVertices[vIndex[0]]
            same = []
            for i in range(0, len(self.geoVertices)):
                if Mesh.comparePoints(self.geoVertices[i], vert):
                    same.append(i)
            last = len(retVert)
            for poly in retPoly:
                for i in range(0, len(poly)):
                    if poly[i] in same:
                        poly[i] = last
            for ind in same:
                vIndex.remove(ind)
            retVert.append(vert)
        self.geoVertices = retVert
        self.geoPolygons = retPoly

    @staticmethod
    def comparePoints(p0, p1):
        def eq(a, b):
            TOLERANCE = 1e-6
            return a - TOLERANCE <= b <= a + TOLERANCE
        return eq(p0[0], p1[0]) and eq(p0[1], p1[1]) and eq(p0[2], p1[2])

    @staticmethod
    def tesselate(patch):
        if len(patch) < 3:
            raise Exception()
        elif len(patch) < 5:
            return [patch]
        else:
            triangles = []
            triangles.extend(map(lambda x: [patch[0], patch[x], patch[x + 1]], range(1, len(patch) - 1)))
            return triangles


class AttributedMesh(Mesh):
    def __init__(self, parent=None, name=None, regions=[]):
        Mesh.__init__(self, parent, name)

        self.regions = {}
        for box, key in regions:
            t = (max(box[0][0], box[1][0]), max(box[0][1], box[1][1]), max(box[0][2], box[1][2]))
            b = (min(box[0][0], box[1][0]), min(box[0][1], box[1][1]), min(box[0][2], box[1][2]))
            key = int(key)
            self.regions[key] = (t, b)
        self.attributes = []

    @staticmethod
    def intersection(region, point):
        # Check whether the point is within the region
        t, b = region[0], region[1]
        return b[0] <= point[0] <= t[0] and b[1] <= point[1] <= t[1] and b[2] <= point[2] <= t[2]

    def associateVertices(self):
        self.attributes = [0 for i in range(0, len(self.geoVertices))]
        for key in self.regions.keys():
            for i in range(0, len(self.geoVertices)):
                if AttributedMesh.intersection(self.regions[key], self.geoVertices[i]):
                    self.attributes[i] = key

    def applyTransforms(self, transforms):
        if len(self.geoVertices) > len(self.attributes):
            raise Exception()
        for i in range(0, len(self.geoVertices)):
            if self.attributes[i] >= len(transforms):
                raise Exception()
            self.geoVertices[i] = transforms[self.attributes[i]].apply(self.geoVertices[i])

    def append(self, other):
        #TODO Optimize
        Mesh.append(self, other)
        self.associateVertices()


class LineArray(Object):
    class Appearance:
        def __init__(self):
            self.material = Material()


    def __init__(self, parent=None, name=None):
        Object.__init__(self, Object.LINES, parent, name)

        if self.parent is None:
            self.geoVertices, self.geoPolygons = [], []
            self.visualAppearance = LineArray.Appearance()

    def appearance(self):
        return self.visualAppearance if self.parent is None else self.parent.appearance()

    def geometry(self):
        return (self.geoVertices, self.geoPolygons) if self.parent is None else self.parent.geometry()

    def append(self, other):
        geoSize = len(self.geoVertices)
        geoVertices, geoPolygons = other.geometry()

        for entry in geoPolygons:
            self.geoPolygons.append([geoSize + vertex for vertex in entry])

        if other.transform is None:
            self.geoVertices += geoVertices
        else:
            self.geoVertices += [other.transform.apply(v) for v in geoVertices]


class Transform:
    def __init__(self):
        self.value = numpy.matrix([
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0]])

    def translate(self, pos):
        mat = numpy.matrix([
                [0.0, 0.0, 0.0, pos[0]],
                [0.0, 0.0, 0.0, pos[1]],
                [0.0, 0.0, 0.0, pos[2]],
                [0.0, 0.0, 0.0,    0.0]])
        self.value = self.value + mat

    def rotate(self, vector, angle):
        mat = rotationMatrix(vector, angle)
        self.value = self.value * mat

    def scale(self, scale):
        mat = numpy.matrix([
                [scale[0],      0.0,      0.0, 0.0],
                [     0.0, scale[1],      0.0, 0.0],
                [     0.0,      0.0, scale[2], 0.0],
                [     0.0,      0.0,      0.0, 1.0]])
        self.value = self.value * mat

    def apply(self, vertex):
        mat = self.value * numpy.matrix([[vertex[0]], [vertex[1]], [vertex[2]], [1.0]])
        return numpy.array(mat)[:,0][0:3]

    def __mul__(self, other):
        transform = copy.deepcopy(self)
        transform.value *= other.value
        return transform

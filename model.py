#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# model.py
# Copyright (C) 2013 xent
# Project is distributed under the terms of the GNU General Public License v3.0

import copy
import math
import numpy

def angle(vect1, vect2):
    norm_vect1 = normalize(vect1[0:3])
    norm_vect2 = normalize(vect2[0:3])
    return math.acos(numpy.clip(numpy.dot(norm_vect1, norm_vect2), -1.0, +1.0))

def normalize(vect):
    length = numpy.linalg.norm(vect)
    return vect / length if length != 0.0 else vect

def tangent(vertex1, vertex2, tangent1, tangent2):
    div = tangent1[1] * tangent2[0] - tangent1[0] * tangent2[1]
    if div != 0.0:
        return (vertex1[0:3] * -tangent2[1] + vertex2[0:3] * tangent1[1]) / div
    return numpy.array([0.0, 0.0, 1.0])

def create_model_view_matrix(eye, center, z_axis):
    forward = normalize(center - eye)
    side = numpy.cross(forward, normalize(z_axis))
    side = numpy.array([0.0, 1.0, 0.0]) if numpy.linalg.norm(side) == 0.0 else normalize(side)
    z_axis = normalize(numpy.cross(side, forward))

    result = numpy.matrix([
        [    1.0,     0.0,     0.0, 0.0],
        [    0.0,     1.0,     0.0, 0.0],
        [    0.0,     0.0,     1.0, 0.0],
        [-eye[0], -eye[1], -eye[2], 1.0]])
    result *= numpy.matrix([
        [side[0], z_axis[0], -forward[0], 0.0],
        [side[1], z_axis[1], -forward[1], 0.0],
        [side[2], z_axis[2], -forward[2], 0.0],
        [    0.0,       0.0,         0.0, 1.0]])
    return result

def create_perspective_matrix(aspect, rotation, distance):
    near, far = distance
    fov = math.radians(rotation) / 4.0
    height = 1.0 / math.tan(fov)
    width = height / aspect

    result = numpy.matrix([
        [width, 0.0, 0.0, 0.0],
        [0.0, height, 0.0, 0.0],
        [0.0, 0.0, -(far + near) / (far - near), -1.0],
        [0.0, 0.0, -(2.0 * far * near) / (far - near), 0.0]])
    return result

def create_orthographic_matrix(area, distance):
    near, far = distance
    width, height = area

    result = numpy.matrix([
        [1.0 / width, 0.0, 0.0, 0.0],
        [0.0, 1.0 / height, 0.0, 0.0],
        [0.0, 0.0, -2.0 / (far - near), 0.0],
        [0.0, 0.0, -(far + near) / (far - near), 1.0]])
    return result

def uv_wrap_planar(mesh, borders=None):
    if borders is None:
        borders = [
            [mesh.geo_vertices[0][0], mesh.geo_vertices[0][1]],
            [mesh.geo_vertices[0][0], mesh.geo_vertices[0][1]]]

        for vert in mesh.geo_vertices:
            if vert[0] < borders[0][0]:
                borders[0][0] = vert[0]
            if vert[0] > borders[1][0]:
                borders[1][0] = vert[0]
            if vert[1] < borders[0][1]:
                borders[0][1] = vert[1]
            if vert[1] > borders[1][1]:
                borders[1][1] = vert[1]

    size = (borders[1][0] - borders[0][0], borders[1][1] - borders[0][1])
    for poly in mesh.geo_polygons:
        for index in poly:
            # pylint: disable=C0103
            u = (mesh.geo_vertices[index][0] - borders[0][0]) / size[0]
            v = (mesh.geo_vertices[index][1] - borders[0][1]) / size[1]
            mesh.tex_vertices.append(numpy.array([u, v]))
            # pylint: enable=C0103
        mesh.tex_polygons.append(poly)

def make_rotation_matrix(vect, rotation):
    cos, sin = math.cos(rotation), math.sin(rotation)

    a11 = cos + vect[0] * vect[0] * (1.0 - cos)
    a12 = vect[0] * vect[1] * (1.0 - cos) - vect[2] * sin
    a13 = vect[0] * vect[2] * (1.0 - cos) + vect[1] * sin
    a21 = vect[1] * vect[0] * (1.0 - cos) + vect[2] * sin
    a22 = cos + vect[1] * vect[1] * (1.0 - cos)
    a23 = vect[1] * vect[2] * (1.0 - cos) - vect[0] * sin
    a31 = vect[2] * vect[0] * (1.0 - cos) - vect[1] * sin
    a32 = vect[2] * vect[1] * (1.0 - cos) + vect[0] * sin
    a33 = cos + vect[2] * vect[2] * (1.0 - cos)

    # Column-major order
    result = numpy.matrix([
        [a11, a12, a13, 0.0],
        [a21, a22, a23, 0.0],
        [a31, a32, a33, 0.0],
        [0.0, 0.0, 0.0, 1.0]])
    return result

def rpy_to_matrix(angles):
    # Roll
    cosr, sinr = math.cos(angles[0]), math.sin(angles[0])
    # Pitch
    cosp, sinp = math.cos(angles[1]), math.sin(angles[1])
    # Yaw
    cosy, siny = math.cos(angles[2]), math.sin(angles[2])

    yaw_matrix = numpy.matrix([
        [ cosy, -siny,   0.0, 0.0],
        [ siny,  cosy,   0.0, 0.0],
        [  0.0,   0.0,   1.0, 0.0],
        [  0.0,   0.0,   0.0, 1.0]])
    pitch_matrix = numpy.matrix([
        [ cosp,   0.0,  sinp, 0.0],
        [  0.0,   1.0,   0.0, 0.0],
        [-sinp,   0.0,  cosp, 0.0],
        [  0.0,   0.0,   0.0, 1.0]])
    roll_matrix = numpy.matrix([
        [  1.0,   0.0,   0.0, 0.0],
        [  0.0,  cosr, -sinr, 0.0],
        [  0.0,  sinr,  cosr, 0.0],
        [  0.0,   0.0,   0.0, 1.0]])
    return yaw_matrix * pitch_matrix * roll_matrix

def quaternion_to_matrix(quaternion):
    # pylint: disable=C0103
    w, x, y, z = quaternion[0], quaternion[1], quaternion[2], quaternion[3]
    matrix = numpy.matrix([
        [
            1.0 - (2.0 * y * y) - (2.0 * z * z),
            (2.0 * x * y) - (2.0 * z * w),
            (2.0 * x * z) + (2.0 * y * w),
            0.0
        ], [
            (2.0 * x * y) + (2.0 * z * w),
            1.0 - (2.0 * x * x) - (2.0 * z * z),
            (2.0 * y * z) - (2.0 * x * w),
            0.0
        ], [
            (2.0 * x * z) - (2.0 * y * w),
            (2.0 * y * z) + (2.0 * x * w),
            1.0 - (2.0 * x * x) - (2.0 * y * y),
            0.0
        ], [
            0.0,
            0.0,
            0.0,
            1.0
        ]])
    # pylint: enable=C0103
    return matrix

def matrix_to_quaternion(matrix):
    matrix = matrix.getA()
    trace = matrix[0][0] + matrix[1][1] + matrix[2][2]

    # pylint: disable=C0103
    if trace > 0:
        s = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (matrix[2][1] - matrix[1][2]) / s
        y = (matrix[0][2] - matrix[2][0]) / s
        z = (matrix[1][0] - matrix[0][1]) / s
    elif matrix[0][0] > matrix[1][1] and matrix[0][0] > matrix[2][2]:
        s = math.sqrt(1.0 + matrix[0][0] - matrix[1][1] - matrix[2][2]) * 2.0
        w = (matrix[2][1] - matrix[1][2]) / s
        x = 0.25 * s
        y = (matrix[0][1] + matrix[1][0]) / s
        z = (matrix[0][2] + matrix[2][0]) / s
    elif matrix[1][1] > matrix[2][2]:
        s = math.sqrt(1.0 + matrix[1][1] - matrix[0][0] - matrix[2][2]) * 2.0
        w = (matrix[0][2] - matrix[2][0]) / s
        x = (matrix[0][1] + matrix[1][0]) / s
        y = 0.25 * s
        z = (matrix[1][2] + matrix[2][1]) / s
    else:
        s = math.sqrt(1.0 + matrix[2][2] - matrix[0][0] - matrix[1][1]) * 2.0
        w = (matrix[1][0] - matrix[0][1]) / s
        x = (matrix[0][2] + matrix[2][0]) / s
        y = (matrix[1][2] + matrix[2][1]) / s
        z = 0.25 * s

    return numpy.array([w, x, y, z])

def slerp(q0, q1, t): # pylint: disable=C0103
    q0, q1 = normalize(q0), normalize(q1)
    dot = numpy.sum(q0 * q1)

    if abs(dot) == 1.0:
        return q0
    if dot < 0.0:
        q1 = -q1
        dot = -dot

    theta0 = math.acos(dot)
    sin_theta0 = math.sin(theta0)
    theta = theta0 * t
    sin_theta = math.sin(theta)

    # pylint: disable=C0103
    s0 = math.cos(theta) - dot * sin_theta / sin_theta0
    s1 = sin_theta / sin_theta0
    # pylint: enable=C0103

    return (s0 * q0) + (s1 * q1)


class Material:
    class Color:
        IDENT = 0

        def __init__(self, name=None):
            self.diffuse = numpy.ones(3)
            self.ambient = numpy.zeros(3)
            self.specular = numpy.zeros(3)
            self.emissive = numpy.zeros(3)
            self.shininess = 0.0
            self.transparency = 0.0
            if name is None:
                self.ident = str(Material.Color.IDENT)
                Material.Color.IDENT += 1
            else:
                self.ident = name

        def __eq__(self, other):
            def eqv(vect1, vect2):
                return (
                    math.isclose(vect1[0], vect2[0])
                    and math.isclose(vect1[1], vect2[1])
                    and math.isclose(vect1[2], vect2[2]))

            if isinstance(other, Material.Color):
                return (
                    math.isclose(self.transparency, other.transparency)
                    and math.isclose(self.shininess, other.shininess)
                    and eqv(self.diffuse, other.diffuse)
                    and eqv(self.ambient, other.ambient)
                    and eqv(self.specular, other.specular)
                    and eqv(self.emissive, other.emissive))
            return False

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
            if isinstance(other, Material.Texture):
                return self.path == other.path
            return False

        def __ne__(self, other):
            return not self == other


    def __init__(self):
        self.color = Material.Color()
        self.diffuse = None
        self.normal = None
        self.specular = None

    def __eq__(self, other):
        if isinstance(other, Material):
            return (
                self.color == other.color
                and self.diffuse == other.diffuse
                and self.normal == other.normal
                and self.specular == other.specular)
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

    def rotate(self, vector, rotation):
        if self.transform is None:
            self.transform = Transform()
        self.transform.rotate(vector, rotation)

    def scale(self, arg):
        if self.transform is None:
            self.transform = Transform()
        self.transform.scale(arg)

    def rename(self, name):
        self.ident = name


class Mesh(Object):
    class Appearance:
        def __init__(self):
            self.material = Material()
            self.normals = False

            self.smooth = False
            self.solid = False
            self.wireframe = False


    def __init__(self, parent=None, name=None):
        super().__init__(Object.PATCHES, parent, name)

        if self.parent is None:
            self.geo_vertices, self.geo_polygons = [], []
            self.tex_vertices, self.tex_polygons = [], []
            self.visual_appearance = Mesh.Appearance()

    def appearance(self):
        return self.parent.appearance() if self.parent is not None else self.visual_appearance

    def geometry(self):
        if self.parent is not None:
            return self.parent.geometry()
        return (self.geo_vertices, self.geo_polygons)

    def texture(self):
        if self.parent is not None:
            return self.parent.texture()
        return (self.tex_vertices, self.tex_polygons)

    def is_textured(self):
        if self.parent is None:
            return bool(self.tex_polygons) and len(self.geo_polygons) == len(self.tex_polygons)
        return self.parent.is_textured()

    def append(self, other):
        geo_count = len(self.geo_vertices)
        geo_vertices, geo_polygons = other.geometry()

        for entry in geo_polygons:
            self.geo_polygons.append([geo_count + vertex for vertex in entry])
        if other.transform is None:
            self.geo_vertices += geo_vertices
        else:
            self.geo_vertices += [other.transform.apply(v) for v in geo_vertices]

        tex_count = len(self.tex_vertices)
        tex_vertices, tex_polygons = other.texture()

        for entry in tex_polygons:
            self.tex_polygons.append([tex_count + vertex for vertex in entry])
        self.tex_vertices += tex_vertices

    def apply(self, transform=None):
        if transform is None:
            transform = self.transform
            self.transform = None
        if transform is not None:
            self.geo_vertices = [transform.apply(v) for v in self.geo_vertices]

    def optimize(self):
        if self.parent is not None:
            return

        #TODO Reduce complexity
        ret_vert = []
        ret_poly = copy.deepcopy(self.geo_polygons)
        vert_index = list(range(0, len(self.geo_vertices)))
        while vert_index:
            vert = self.geo_vertices[vert_index[0]]
            same = []
            for i in range(0, len(self.geo_vertices)):
                if Mesh.isclose(self.geo_vertices[i], vert):
                    same.append(i)
            last = len(ret_vert)
            for poly in ret_poly:
                for i, value in enumerate(poly):
                    if value in same:
                        poly[i] = last
            for ind in same:
                vert_index.remove(ind)
            ret_vert.append(vert)
        self.geo_vertices = ret_vert
        self.geo_polygons = ret_poly

    @staticmethod
    def isclose(vect1, vect2):
        # Default relative tolerance 1e-9 is used
        return (
            math.isclose(vect1[0], vect2[0])
            and math.isclose(vect1[1], vect2[1])
            and math.isclose(vect1[2], vect2[2]))

    @staticmethod
    def triangulate(patch):
        if len(patch) < 3:
            raise ValueError('not enough vertices')

        if len(patch) < 5:
            return [patch]
        return [[patch[0], patch[i], patch[i + 1]] for i in range(1, len(patch) - 1)]


class AttributedMesh(Mesh):
    def __init__(self, parent=None, name=None, regions=None):
        super().__init__(parent, name)

        self.regions = {}
        if regions is not None:
            for box, key in regions:
                top = numpy.maximum(box[0], box[1])
                bottom = numpy.minimum(box[0], box[1])
                self.regions[int(key)] = (top, bottom)
        self.attributes = []

    @staticmethod
    def intersection(region, point):
        # Check whether the point is within the region
        top, bottom = region[0], region[1]
        return (
            bottom[0] <= point[0] <= top[0]
            and bottom[1] <= point[1] <= top[1]
            and bottom[2] <= point[2] <= top[2])

    def associate_vertices(self):
        self.attributes = [0 for i in range(0, len(self.geo_vertices))]
        for key in self.regions:
            for i in range(0, len(self.geo_vertices)):
                if AttributedMesh.intersection(self.regions[key], self.geo_vertices[i]):
                    self.attributes[i] = key

    def apply_transform(self, transforms):
        if len(self.geo_vertices) > len(self.attributes):
            raise Exception()
        for i in range(0, len(self.geo_vertices)):
            if self.attributes[i] >= len(transforms):
                raise Exception()
            self.geo_vertices[i] = transforms[self.attributes[i]].apply(self.geo_vertices[i])

    def append(self, other):
        #TODO Optimize
        Mesh.append(self, other)
        self.associate_vertices()


class LineArray(Object):
    class Appearance:
        def __init__(self):
            self.material = Material()


    def __init__(self, parent=None, name=None):
        super().__init__(Object.LINES, parent, name)

        if self.parent is None:
            self.geo_vertices, self.geo_polygons = [], []
            self.visual_appearance = LineArray.Appearance()

    def appearance(self):
        return self.parent.appearance() if self.parent is not None else self.visual_appearance

    def geometry(self):
        if self.parent is not None:
            return self.parent.geometry()
        return (self.geo_vertices, self.geo_polygons)

    def append(self, other):
        geo_count = len(self.geo_vertices)
        geo_vertices, geo_polygons = other.geometry()

        for entry in geo_polygons:
            self.geo_polygons.append([geo_count + vertex for vertex in entry])

        if other.transform is None:
            self.geo_vertices += geo_vertices
        else:
            self.geo_vertices += [other.transform.apply(v) for v in geo_vertices]

    def optimize(self):
        pass # TODO Optimize


class Transform:
    def __init__(self, matrix=None, quaternion=None):
        if matrix is not None:
            self.matrix = matrix
        elif quaternion is not None:
            self.matrix = quaternion_to_matrix(quaternion)
        else:
            self.matrix = numpy.identity(4)

    def translate(self, pos):
        matrix = numpy.matrix([
            [0.0, 0.0, 0.0, pos[0]],
            [0.0, 0.0, 0.0, pos[1]],
            [0.0, 0.0, 0.0, pos[2]],
            [0.0, 0.0, 0.0,    0.0]])
        self.matrix = self.matrix + matrix

    def rotate(self, vector, rotation):
        matrix = make_rotation_matrix(vector, rotation)
        self.matrix = self.matrix * matrix

    def scale(self, scale):
        matrix = numpy.matrix([
            [scale[0],      0.0,      0.0, 0.0],
            [     0.0, scale[1],      0.0, 0.0],
            [     0.0,      0.0, scale[2], 0.0],
            [     0.0,      0.0,      0.0, 1.0]])
        self.matrix = self.matrix * matrix

    def apply(self, vertex):
        matrix = self.matrix * numpy.matrix([[vertex[0]], [vertex[1]], [vertex[2]], [1.0]])
        return numpy.array(matrix)[:,0][0:3]

    def quaternion(self):
        return matrix_to_quaternion(self.matrix)

    def __mul__(self, other):
        transform = Transform()
        transform.matrix = self.matrix * other.matrix
        return transform

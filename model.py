#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# model.py
# Copyright (C) 2013 xent
# Project is distributed under the terms of the GNU General Public License v3.0

import copy
import math
import numpy

def angle(vector_a, vector_b):
    norm_vector_a = normalize(vector_a[0:3])
    norm_vector_b = normalize(vector_b[0:3])
    return math.acos(numpy.clip(numpy.dot(norm_vector_a, norm_vector_b), -1.0, +1.0))

def normalize(vector):
    length = numpy.linalg.norm(vector)
    return vector / length if length != 0.0 else vector

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

    result = numpy.array([
        [    1.0,     0.0,     0.0, 0.0],
        [    0.0,     1.0,     0.0, 0.0],
        [    0.0,     0.0,     1.0, 0.0],
        [-eye[0], -eye[1], -eye[2], 1.0]])
    result = numpy.matmul(result, numpy.array([
        [side[0], z_axis[0], -forward[0], 0.0],
        [side[1], z_axis[1], -forward[1], 0.0],
        [side[2], z_axis[2], -forward[2], 0.0],
        [    0.0,       0.0,         0.0, 1.0]]))
    return result

def create_perspective_matrix(aspect, rotation, distance):
    near, far = distance
    fov = math.radians(rotation) / 4.0
    height = 1.0 / math.tan(fov)
    width = height / aspect

    result = numpy.array([
        [width,    0.0,                                0.0,  0.0],
        [  0.0, height,                                0.0,  0.0],
        [  0.0,    0.0,       -(far + near) / (far - near), -1.0],
        [  0.0,    0.0, -(2.0 * far * near) / (far - near),  0.0]])
    return result

def create_orthographic_matrix(area, distance):
    near, far = distance
    width, height = area

    result = numpy.array([
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

def make_rotation_matrix(vector, rotation):
    cos, sin = math.cos(rotation), math.sin(rotation)

    a11 = cos + vector[0] * vector[0] * (1.0 - cos)
    a12 = vector[0] * vector[1] * (1.0 - cos) - vector[2] * sin
    a13 = vector[0] * vector[2] * (1.0 - cos) + vector[1] * sin
    a21 = vector[1] * vector[0] * (1.0 - cos) + vector[2] * sin
    a22 = cos + vector[1] * vector[1] * (1.0 - cos)
    a23 = vector[1] * vector[2] * (1.0 - cos) - vector[0] * sin
    a31 = vector[2] * vector[0] * (1.0 - cos) - vector[1] * sin
    a32 = vector[2] * vector[1] * (1.0 - cos) + vector[0] * sin
    a33 = cos + vector[2] * vector[2] * (1.0 - cos)

    # Column-major order
    matrix = numpy.array([
        [a11, a12, a13, 0.0],
        [a21, a22, a23, 0.0],
        [a31, a32, a33, 0.0],
        [0.0, 0.0, 0.0, 1.0]])
    return matrix

def reset_allocator():
    Object.IDENT = 0
    Material.Color.IDENT = 0
    Material.Texture.IDENT = 0

def rpy_to_matrix(angles):
    # Roll
    cosr, sinr = math.cos(angles[0]), math.sin(angles[0])
    # Pitch
    cosp, sinp = math.cos(angles[1]), math.sin(angles[1])
    # Yaw
    cosy, siny = math.cos(angles[2]), math.sin(angles[2])

    # Column-major order
    yaw_matrix = numpy.array([
        [ cosy, -siny,   0.0, 0.0],
        [ siny,  cosy,   0.0, 0.0],
        [  0.0,   0.0,   1.0, 0.0],
        [  0.0,   0.0,   0.0, 1.0]])
    pitch_matrix = numpy.array([
        [ cosp,   0.0,  sinp, 0.0],
        [  0.0,   1.0,   0.0, 0.0],
        [-sinp,   0.0,  cosp, 0.0],
        [  0.0,   0.0,   0.0, 1.0]])
    roll_matrix = numpy.array([
        [  1.0,   0.0,   0.0, 0.0],
        [  0.0,  cosr, -sinr, 0.0],
        [  0.0,  sinr,  cosr, 0.0],
        [  0.0,   0.0,   0.0, 1.0]])

    return numpy.matmul(yaw_matrix, numpy.matmul(pitch_matrix, roll_matrix))

def quaternion_to_matrix(quaternion):
    # pylint: disable=C0103
    xx = quaternion[1] * quaternion[1]
    xy = quaternion[1] * quaternion[2]
    xz = quaternion[1] * quaternion[3]
    xw = quaternion[1] * quaternion[0]
    yy = quaternion[2] * quaternion[2]
    yz = quaternion[2] * quaternion[3]
    yw = quaternion[2] * quaternion[0]
    zz = quaternion[3] * quaternion[3]
    zw = quaternion[3] * quaternion[0]

    m00 = 1.0 - 2.0 * (yy + zz)
    m01 =       2.0 * (xy - zw)
    m02 =       2.0 * (xz + yw)

    m10 =       2.0 * (xy + zw)
    m11 = 1.0 - 2.0 * (xx + zz)
    m12 =       2.0 * (yz - xw)

    m20 =       2.0 * (xz - yw)
    m21 =       2.0 * (yz + xw)
    m22 = 1.0 - 2.0 * (xx + yy)

    # Column-major order
    matrix = numpy.array([
        [m00, m01, m02, 0.0],
        [m10, m11, m12, 0.0],
        [m20, m21, m22, 0.0],
        [0.0, 0.0, 0.0, 1.0]])
    # pylint: enable=C0103

    return matrix

def matrix_to_quaternion(matrix):
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
    quaternion = numpy.array([w, x, y, z])
    # pylint: enable=C0103

    return quaternion

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
            self.geo_vertices += [other.transform.apply(vertex) for vertex in geo_vertices]

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
            self.geo_vertices = [transform.apply(vertex) for vertex in self.geo_vertices]

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
            for i, value in enumerate(self.geo_vertices):
                if Mesh.isclose(value, vert):
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
    def isclose(vector_a, vector_b):
        # Default relative tolerance 1e-9 is used
        return (
            math.isclose(vector_a[0], vector_b[0])
            and math.isclose(vector_a[1], vector_b[1])
            and math.isclose(vector_a[2], vector_b[2]))

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
        self.attributes = [0] * len(self.geo_vertices)
        for key, region in self.regions.items():
            for i, value in enumerate(self.geo_vertices):
                if AttributedMesh.intersection(region, value):
                    self.attributes[i] = key

    def apply_transform(self, transforms):
        if len(self.geo_vertices) > len(self.attributes):
            raise Exception()
        for i, value in enumerate(self.geo_vertices):
            if self.attributes[i] >= len(transforms):
                raise Exception()
            self.geo_vertices[i] = transforms[self.attributes[i]].apply(value)

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
            self.geo_vertices += [other.transform.apply(vertex) for vertex in geo_vertices]

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
        # Column-major order
        matrix = numpy.array([
            [0.0, 0.0, 0.0, pos[0]],
            [0.0, 0.0, 0.0, pos[1]],
            [0.0, 0.0, 0.0, pos[2]],
            [0.0, 0.0, 0.0,    0.0]])
        self.matrix = self.matrix + matrix

    def rotate(self, vector, rotation):
        matrix = make_rotation_matrix(vector, rotation)
        self.matrix = numpy.matmul(self.matrix, matrix)

    def scale(self, scale):
        matrix = numpy.array([
            [scale[0],      0.0,      0.0, 0.0],
            [     0.0, scale[1],      0.0, 0.0],
            [     0.0,      0.0, scale[2], 0.0],
            [     0.0,      0.0,      0.0, 1.0]])
        self.matrix = numpy.matmul(self.matrix, matrix)

    def apply(self, vertex):
        return numpy.matmul(self.matrix, numpy.array([*vertex, 1.0]))[0:3]

    def quaternion(self):
        return matrix_to_quaternion(self.matrix)

    def __mul__(self, other):
        transform = Transform()
        transform.matrix = numpy.matmul(self.matrix, other.matrix)
        return transform

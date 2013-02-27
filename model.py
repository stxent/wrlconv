#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# model.py
# Copyright (C) 2013 xent
# Project is distributed under the terms of the GNU General Public License v3.0

import numpy
import math


class Material:
    class Color:
        def __init__(self):
            self.diffuse = [1., 1., 1., 1.]
            self.ambient = [1., 1., 1., 1.]
            self.specular = [0., 0., 0., 1.]
            self.emissive = [0., 0., 0., 1.]
            self.shininess = 0.
            self.transparency = 0.

    class Texture:
        def __init__(self):
            self.name = ""

    def __init__(self):
        self.shader = ""


class Transform:
    def __init__(self):
        self.value = numpy.matrix([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]])

    def translate(self, pos):
        mat = numpy.matrix([[0., 0., 0., pos[0]], [0., 0., 0., pos[1]], [0., 0., 0., pos[2]], [0., 0., 0., 0.]])
        self.value = self.value + mat

    def rotate(self, angles):
        cs = math.cos(angle)
        sn = math.sin(angle)
        a = numpy.array([angles[0], angles[1], angles[2]]) #FIXME float?
        mat = numpy.matrix([[cs + a[0] * a[0] * (1 - cs), a[0] * a[1] * (1 - cs) - a[2] * sn,
                             a[0] * a[2] * (1 - cs) + a[1] * sn, 0.],
                            [a[1] * a[0] * (1 - cs) + a[2] * sn, cs + a[1] * a[1] * (1 - cs),
                             a[1] * a[2] * (1 - cs) - a[0] * sn, 0.],
                            [a[2] * a[0] * (1 - cs) - a[1] * sn, a[2] * a[1] * (1 - cs) + a[0] * sn,
                             cs + a[2] * a[2] * (1 - cs), 0.],
                            [0., 0., 0., 1.]])
        self.value = self.value * mat

    def scale(self, value):
        mat = numpy.matrix([[value[0], 0., 0., 0.], [0., value[1], 0., 0.], [0., 0., value[2], 0.], [0., 0., 0., 1.]])
        self.value = self.value * scale


class Mesh:
    def __init__(self, parent = None):
        self.transform = None
        self.parent = parent

        if self.parent is None:
            self.vertices = []
            self.polygons = []
            self.texels = []
            self.material = Material()
        else:
            self.vertices = parent.vertices
            self.polygons = parent.polygons
            self.texels = parent.texels
            self.material = parent.material #FIXME Separate material?

    def append(self, other):
        size = len(self.vertices)
        for poly in other.polygons:
            newPoly = [] #FIXME Optimize
            for index in poly:
                newPoly.append(index + size)
            self.polygons.append(newPoly)
        if other.transform is None:
            self.vertices.extend(other.vertices)
        else:
            for vert in other.vertices:
                tmp = other.transform.value * numpy.matrix([[vert[0]], [vert[1]], [vert[2]], [1.]])
                self.vertices.append(numpy.array([tmp[0], tmp[1], tmp[2]]))

    def translate(self, arg):
        if self.transform is None:
            self.transform = Transform()
        self.transform.translate(arg)

    def rotate(self, arg):
        if self.transform is None:
            self.transform = Transform()
        self.transform.rotate(arg)

    def scale(self, arg):
        if self.transform is None:
            self.transform = Transform()
        self.transform.scale(arg)

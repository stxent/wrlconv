#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# helpers.py
# Copyright (C) 2016 xent
# Project is distributed under the terms of the GNU General Public License v3.0

import math
import numpy as np

try:
    import geometry
    import model
except ImportError:
    from . import geometry
    from . import model

def make_blue_material():
    mat = model.Material()
    mat.color.diffuse = np.array([0.0, 0.0, 1.0])
    return mat

def make_green_material():
    mat = model.Material()
    mat.color.diffuse = np.array([0.0, 1.0, 0.0])
    return mat

def make_red_material():
    mat = model.Material()
    mat.color.diffuse = np.array([1.0, 0.0, 0.0])
    return mat

def make_dark_gray_material():
    mat = model.Material()
    mat.color.diffuse = np.array([0.3, 0.3, 0.3])
    return mat

def make_light_gray_material():
    mat = model.Material()
    mat.color.diffuse = np.array([0.5, 0.5, 0.5])
    return mat

def make_axes(length=4.0):
    # Objects
    x_axis = model.LineArray(name='XAxisHelper')
    x_axis.appearance().material = make_red_material()
    x_axis.geo_vertices.extend([np.zeros(3), np.array([length, 0.0, 0.0])])
    x_axis.geo_polygons.append([0, 1])
    y_axis = model.LineArray(name='YAxisHelper')
    y_axis.appearance().material = make_green_material()
    y_axis.geo_vertices.extend([np.zeros(3), np.array([0.0, length, 0.0])])
    y_axis.geo_polygons.append([0, 1])
    z_axis = model.LineArray(name='ZAxisHelper')
    z_axis.appearance().material = make_blue_material()
    z_axis.geo_vertices.extend([np.zeros(3), np.array([0.0, 0.0, length / 2.0])])
    z_axis.geo_polygons.append([0, 1])

    return [x_axis, y_axis, z_axis]

def make_grid():
    # Materials
    dark_material = make_dark_gray_material()
    light_material = make_light_gray_material()

    # Objects
    z_grid = geometry.Plane((10, 10), (10, 10))
    z_grid.appearance().material = dark_material
    z_grid.appearance().wireframe = True
    z_grid.rename('ZGridHelper')
    x_grid = geometry.Plane((2, 10), (2, 10))
    x_grid.appearance().material = light_material
    x_grid.appearance().wireframe = True
    x_grid.rotate([0.0, 1.0, 0.0], math.pi / 2.0)
    x_grid.rename('XGridHelper')
    y_grid = geometry.Plane((10, 2), (10, 2))
    y_grid.appearance().material = light_material
    y_grid.appearance().wireframe = True
    y_grid.rotate([1.0, 0.0, 0.0], math.pi / 2.0)
    y_grid.rename('YGridHelper')

    return [x_grid, y_grid, z_grid]

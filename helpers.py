#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# helpers.py
# Copyright (C) 2016 xent
# Project is distributed under the terms of the GNU General Public License v3.0

import math
import numpy

import geometry
import model

def createAxes(length=4.):
    #Materials
    redMaterial = model.Material()
    redMaterial.color.diffuse = numpy.array([1., 0., 0.])
    greenMaterial = model.Material()
    greenMaterial.color.diffuse = numpy.array([0., 1., 0.])
    blueMaterial = model.Material()
    blueMaterial.color.diffuse = numpy.array([0., 0., 1.])

    #Objects
    xAxis = model.LineArray(name="xAxisHelper")
    xAxis.visualAppearance.material = redMaterial
    xAxis.geoVertices.extend([numpy.array([0., 0., 0.]), numpy.array([length, 0., 0.])])
    xAxis.geoPolygons.append([0, 1])
    yAxis = model.LineArray(name="yAxisHelper")
    yAxis.visualAppearance.material = greenMaterial
    yAxis.geoVertices.extend([numpy.array([0., 0., 0.]), numpy.array([0., length, 0.])])
    yAxis.geoPolygons.append([0, 1])
    zAxis = model.LineArray(name="zAxisHelper")
    zAxis.visualAppearance.material = blueMaterial
    zAxis.geoVertices.extend([numpy.array([0., 0., 0.]), numpy.array([0., 0., length / 2.])])
    zAxis.geoPolygons.append([0, 1])

    return [xAxis, yAxis, zAxis]

def createGrid():
    #Materials
    darkGrayMaterial = model.Material()
    darkGrayMaterial.color.diffuse = numpy.array([0.3] * 3)
    lightGrayMaterial = model.Material()
    lightGrayMaterial.color.diffuse = numpy.array([0.5] * 3)

    #Objects
    zGrid = geometry.Plane((10, 10), (10, 10))
    zGrid.visualAppearance.material = darkGrayMaterial
    zGrid.visualAppearance.wireframe = True
    xGrid = geometry.Plane((2, 10), (2, 10))
    xGrid.visualAppearance.material = lightGrayMaterial
    xGrid.visualAppearance.wireframe = True
    xGrid.rotate([0., 1., 0.], math.pi / 2.)
    yGrid = geometry.Plane((10, 2), (10, 2))
    yGrid.visualAppearance.material = lightGrayMaterial
    yGrid.visualAppearance.wireframe = True
    yGrid.rotate([1., 0., 0.], math.pi / 2.)

    return [xGrid, yGrid, zGrid]

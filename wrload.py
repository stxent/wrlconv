#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# wrload.py
# Copyright (C) 2011 xent
# Project is distributed under the terms of the GNU General Public License v3.0

import argparse
import math
import numpy
import os
import re

import geometry
import model
import render_ogl41
import vrml_export
import vrml_export_kicad
import vrml_import
import x3d_export
import x3d_import

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

def createAxes():
    #Materials
    redMaterial = model.Material()
    redMaterial.color.diffuse = numpy.array([1., 0., 0.])
    greenMaterial = model.Material()
    greenMaterial.color.diffuse = numpy.array([0., 1., 0.])
    blueMaterial = model.Material()
    blueMaterial.color.diffuse = numpy.array([0., 0., 1.])
    #Objects
    length = 4.
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

parser = argparse.ArgumentParser()
parser.add_argument("-v", dest="view", help="render model", default=False, action="store_true")
parser.add_argument("-o", dest="output", help="write model to specified file", default="")
parser.add_argument("-t", dest="translation", help="move mesh to new coordinates x,y,z", default='0.,0.,0.')
parser.add_argument("-r", dest="rotation", help="rotate mesh around vector x,y,z by angle in degrees",\
        default='0.,0.,1.,0.')
parser.add_argument("-s", dest="scale", help="scale shapes by x,y,z",\
        default='1.,1.,1.')
parser.add_argument("-f", dest="pattern", help="regular expression, filter objects by name", default="")
parser.add_argument("-d", dest="debug", help="show debug information", default=False, action="store_true")
parser.add_argument("--kicad", dest="kicad", help="export to KiCad with simplified syntax",\
        default=False, action="store_true")
parser.add_argument("--axes", dest="axes", help="show axes", default=False, action="store_true")
parser.add_argument("--grid", dest="grid", help="show grid", default=False, action="store_true")
parser.add_argument("--normals", dest="normals", help="show normals", default=False, action="store_true")
parser.add_argument(dest="files", nargs="*")
options = parser.parse_args()

globalRotation = [0., 0., 1., 0.]
globalScale = [1., 1., 1.]
globalTranslation = [0., 0., 0.]

try:
    globalRotation = map(float, options.rotation.split(","))[0:4]
    globalScale = map(float, options.scale.split(","))[0:3]
    globalTranslation = map(float, options.translation.split(","))[0:3]
except ValueError:
    print "Wrong argument"
    exit()

globalRotation[3] *= math.pi / 180.

exportList = []

if options.debug:
    render_ogl41.debugEnabled = True
    vrml_export.debugEnabled = True
    vrml_export_kicad.debugEnabled = True
    vrml_import.debugEnabled = True
    x3d_import.debugEnabled = True
    x3d_export.debugEnabled = True

for filename in options.files:
    extension = os.path.splitext(filename)[1][1:].lower()
    if extension == "wrl":
        exportList.extend(vrml_import.load(filename))
    elif extension == "x3d":
        exportList.extend(x3d_import.load(filename))

for mesh in exportList:
    mesh.transform.rotate(globalRotation[0:3], globalRotation[3])
    mesh.transform.scale(globalScale)
    mesh.transform.translate(globalTranslation)

    if options.normals:
        mesh.appearance().normals = True

if options.pattern != "":
    exportList = filter(lambda mesh: re.search(options.pattern, mesh.ident, re.S) is not None, exportList)

if options.output != "":
    extension = os.path.splitext(options.output)[1][1:].lower()
    if extension == "wrl":
        if options.kicad:
            vrml_export_kicad.store(exportList, options.output)
        else:
            vrml_export.store(exportList, options.output, vrml_export.VRML_STRICT)
    elif extension == "x3d":
        x3d_export.store(exportList, options.output)

if options.view:
    helpers = []
    if options.grid:
        helpers += createGrid()
    if options.axes:
        helpers += createAxes()
    render = render_ogl41.Render(exportList + helpers, {"overlay": True, "antialiasing": 4})

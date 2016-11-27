#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# generator_lqfp.py
# Copyright (C) 2015 xent
# Project is distributed under the terms of the GNU General Public License v3.0

import argparse
import copy
import math
import os
import re

import helpers
import model
import render_ogl41
import vrml_export
import vrml_export_kicad
import vrml_import
import x3d_export
import x3d_import

#TODO Add input file option
parser = argparse.ArgumentParser()
parser.add_argument("-v", dest="view", help="render models", default=False, action="store_true")
parser.add_argument("-o", dest="output", help="write models to specified directory", default="")
parser.add_argument("-f", dest="format", help="output file format", default="x3d")
parser.add_argument("-d", dest="debug", help="show debug information", default=False, action="store_true")
parser.add_argument("--fast", dest="fast", help="disable visual effects", default=False, action="store_true")
parser.add_argument(dest="files", nargs="*")
options = parser.parse_args()

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

def lookup(meshList, meshName):
    found = []
    for entry in meshList:
        if re.search(meshName, entry.ident, re.S) is not None:
            found.append(entry)
    return found

lqfpBody = lookup(exportList, "PatLQFPBody")[0].parent
lqfpBodyHole = lookup(exportList, "PatLQFPBody")[1].parent
lqfpNarrowPin = lookup(exportList, "PatLQFPNarrowPin")[0].parent
lqfpWidePin = lookup(exportList, "PatLQFPWidePin")[0].parent

#Modified SO models
lqfpRegions = [
        (((0.5, 0.5, 1.0), (-0.5, -0.5, -1.0)), 1),
        (((1.5, 1.5, 1.0), (0.5, 0.5, -1.0)), 2),
        (((-1.5, 1.5, 1.0), (-0.5, 0.5, -1.0)), 3),
        (((1.5, -1.5, 1.0), (0.5, -0.5, -1.0)), 4),
        (((-1.5, -1.5, 1.0), (-0.5, -0.5, -1.0)), 5)]
lqfpAttributedBody = model.AttributedMesh(name="LQFPBody", regions=lqfpRegions)
lqfpAttributedBody.append(lqfpBody)
lqfpAttributedBody.visualAppearance = lqfpBody.appearance()

#Body model, pin model, pin spacing
narrowPattern = (lqfpAttributedBody, lqfpBodyHole, lqfpNarrowPin, 0.5)
widePattern = (lqfpAttributedBody, lqfpBodyHole, lqfpWidePin, 0.8)

def createBody(pattern, width, count, name):
    DEFAULT_WIDTH = 5.0
    bodyDelta = (width - DEFAULT_WIDTH) / 2. / 2.54
    margin = width / 2. / 2.54
    spacing = pattern[3] / 2.54

    sideCount = count / 4
    offset = spacing / 2. if sideCount % 2 == 0 else spacing
    dot = (-sideCount / 2. + 1.5) * spacing - offset

    corners = [model.Transform(), model.Transform(),
            model.Transform(), model.Transform()]
    center = model.Transform()
    center.translate([dot, dot, 0.])
    corners[0].translate([bodyDelta, bodyDelta, 0.])
    corners[1].translate([-bodyDelta, bodyDelta, 0.])
    corners[2].translate([bodyDelta, -bodyDelta, 0.])
    corners[3].translate([-bodyDelta, -bodyDelta, 0.])
    transforms = [model.Transform()] + [center] + corners
    body = copy.deepcopy(pattern[0])
    body.applyTransforms(transforms)
    body.translate([0., 0., 0.001])
    
    hole = copy.deepcopy(pattern[1])
    hole.translate([dot, dot, 0.001])

    def makePin(x, y, angle, number):
        pin = model.Mesh(parent=pattern[2], name="%s%uPin%u" % (name, count, number))
        pin.translate([x, y, 0.001])
        pin.rotate([0., 0., 1.], angle * math.pi / 180.)
        return pin

    pins = []
    for i in range(0, sideCount):
        x = (i - sideCount / 2 + 1) * spacing - offset
        y = margin

        pins.append(makePin(x, y, 180., i + 1))
        pins.append(makePin(-x, -y, 0., i + 1 + sideCount * 2))
        x, y = y, x
        pins.append(makePin(x, -y, 90., i + 1 + sideCount))
        pins.append(makePin(-x, y, -90., i + 1 + sideCount * 3))

    return [body, hole] + pins

models = []

models.append((createBody(widePattern, 7.0, 32, "LQFP"), "lqfp32"))
models.append((createBody(narrowPattern, 7.0, 48, "LQFP"), "lqfp48"))
models.append((createBody(narrowPattern, 10.0, 64, "LQFP"), "lqfp64"))
models.append((createBody(narrowPattern, 12.0, 80, "LQFP"), "lqfp80"))
models.append((createBody(narrowPattern, 14.0, 100, "LQFP"), "lqfp100"))
models.append((createBody(narrowPattern, 20.0, 144, "LQFP"), "lqfp144"))
models.append((createBody(narrowPattern, 24.0, 176, "LQFP"), "lqfp176"))

if options.output != "":
    outputPath = options.output
    if outputPath[-1] != '/':
        outputPath += '/'
    if options.format == "wrl":
        for entry in models:
            vrml_export_kicad.store(entry[0], outputPath + entry[1] + ".wrl")
    elif options.format == "x3d":
        for entry in models:
            x3d_export.store(entry[0], outputPath + entry[1] + ".x3d")
    else:
        raise Exception()

if options.view:
    effects = {} if options.fast else {"antialiasing": 4}
    helperObjects = helpers.createGrid()
    exportList = []
    [exportList.extend(entry[0]) for entry in models]
    render = render_ogl41.Render(helperObjects + exportList, effects)

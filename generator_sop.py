#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# generator_sop.py
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

soBody = lookup(exportList, "PatSOBody")[0].parent
soBodyHole = lookup(exportList, "PatSOBody")[1].parent
soPin = lookup(exportList, "PatSOPin")[0].parent

tssopBody = lookup(exportList, "PatTSSOPBody")[0].parent
tssopBodyHole = lookup(exportList, "PatTSSOPBody")[1].parent
tssopPin = lookup(exportList, "PatTSSOPPin")[0].parent

#Modified SO models
soRegions = [
        (((0.15, 0.15, 1.0), (-0.15, -0.15, -1.0)), 1),
        (((1.0, 1.0, 1.0), (0.15, 0.15, -1.0)), 2),
        (((-1.0, 1.0, 1.0), (-0.15, 0.15, -1.0)), 3),
        (((1.0, -1.0, 1.0), (0.15, -0.15, -1.0)), 4),
        (((-1.0, -1.0, 1.0), (-0.15, -0.15, -1.0)), 5)]
soAttributedBody = model.AttributedMesh(name="SOBody", regions=soRegions)
soAttributedBody.append(soBody)
soAttributedBody.visualAppearance = soBody.appearance()

#TSSOP model uses same regions
tssopAttributedBody = model.AttributedMesh(name="TSSOPBody", regions=soRegions)
tssopAttributedBody.append(tssopBody)
tssopAttributedBody.visualAppearance = tssopBody.appearance()

#Body model, pin model, pin spacing, dot edge offset
soPattern = (soAttributedBody, soBodyHole, soPin, 1.27, 0.25)
tssopPattern = (tssopAttributedBody, tssopBodyHole, tssopPin, 0.65, 0.0)

def createBody(pattern, size, count, name):
    DEFAULT_WIDTH = 2.0
    bodyDelta = ((size[0] - DEFAULT_WIDTH) / 2. / 2.54, (size[1] - DEFAULT_WIDTH) / 2. / 2.54)
    margin = size[1] / 2. / 2.54
    spacing = pattern[3] / 2.54

    sideCount = count / 2
    offset = spacing / 2. if sideCount % 2 == 0 else spacing
    dot = (-((size[0] / 2. - 0.75) / 2.54), -((size[1] / 2. - 0.75 - pattern[4]) / 2.54))

    corners = [model.Transform(), model.Transform(),
            model.Transform(), model.Transform()]
    center = model.Transform()
    center.translate([dot[0], dot[1], 0.])
    corners[0].translate([bodyDelta[0], bodyDelta[1], 0.])
    corners[1].translate([-bodyDelta[0], bodyDelta[1], 0.])
    corners[2].translate([bodyDelta[0], -bodyDelta[1], 0.])
    corners[3].translate([-bodyDelta[0], -bodyDelta[1], 0.])
    transforms = [model.Transform()] + [center] + corners
    body = copy.deepcopy(pattern[0])
    body.applyTransforms(transforms)
    body.translate([0., 0., 0.001])
    
    hole = copy.deepcopy(pattern[1])
    hole.translate([dot[0], dot[1], 0.001])

    def makePin(x, y, angle, number):
        pin = model.Mesh(parent=pattern[2], name="%s%uPin%u" % (name, count, number))
        pin.translate([x, y, 0.001])
        pin.rotate([0., 0., 1.], angle * math.pi / 180.)
        return pin

    pins = []
    for i in range(0, sideCount):
        x = (i - sideCount / 2 + 1) * spacing - offset
        y = margin

        pins.append(makePin(x, y, 180., i + 1 + sideCount))
        pins.append(makePin(-x, -y, 0., i + 1))

    return [body, hole] + pins

models = []

models.append((createBody(soPattern, (10.1, 7.4), 16, "SO16W"), "so16w"))
models.append((createBody(soPattern, (11.35, 7.4), 18, "SO18W"), "so18w"))
models.append((createBody(soPattern, (12.6, 7.4), 20, "SO20W"), "so20w"))
models.append((createBody(soPattern, (15.2, 7.4), 24, "SO24W"), "so24w"))
models.append((createBody(soPattern, (17.7, 7.4), 28, "SO28W"), "so28w"))

models.append((createBody(soPattern, (4.8, 3.7), 8, "SO8N"), "so8n"))
models.append((createBody(soPattern, (8.55, 3.7), 14, "SO14N"), "so14n"))
models.append((createBody(soPattern, (9.8, 3.7), 16, "SO16N"), "so16n"))

models.append((createBody(tssopPattern, (2.9, 4.3), 8, "TSSOP8"), "tssop8"))
models.append((createBody(tssopPattern, (4.9, 4.3), 14, "TSSOP14"), "tssop14"))
models.append((createBody(tssopPattern, (4.9, 4.3), 16, "TSSOP16"), "tssop16"))
models.append((createBody(tssopPattern, (6.4, 4.3), 20, "TSSOP20"), "tssop20"))
models.append((createBody(tssopPattern, (7.7, 4.3), 24, "TSSOP24"), "tssop24"))
models.append((createBody(tssopPattern, (9.6, 4.3), 28, "TSSOP28"), "tssop28"))

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

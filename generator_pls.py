#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# generator_pls.py
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
    for entry in meshList:
        if re.search(meshName, entry.ident, re.S) is not None:
            return entry
    raise Exception()

plsBody = lookup(exportList, "PatPLSBody").parent
plsEdgeBody = lookup(exportList, "PatPLSEdgeBody").parent
plsPin = lookup(exportList, "PatPLSPin").parent
plsJumper = lookup(exportList, "PatPLSJumper").parent

pldBody = lookup(exportList, "PatPLDBody").parent
pldEdgeBody = lookup(exportList, "PatPLDEdgeBody").parent
pldPin = lookup(exportList, "PatPLDPin").parent

pls2Body = lookup(exportList, "PatPLS2Body").parent
pls2EdgeBody = lookup(exportList, "PatPLS2EdgeBody").parent
pls2Pin = lookup(exportList, "PatPLS2Pin").parent
pls2Jumper = lookup(exportList, "PatPLS2Jumper").parent

pld2Body = lookup(exportList, "PatPLD2Body").parent
pld2EdgeBody = lookup(exportList, "PatPLD2EdgeBody").parent
pld2Pin = lookup(exportList, "PatPLD2Pin").parent

bhBody = lookup(exportList, "PatBHBody").parent
bhPin = lookup(exportList, "PatBHPin").parent

#Body model, edge model, pin model, spacing, number of rows
plsPattern = (plsBody, plsEdgeBody, plsPin, 1.0, 1)
pldPattern = (pldBody, pldEdgeBody, pldPin, 1.0, 2)
pls2Pattern = (pls2Body, pls2EdgeBody, pls2Pin, 0.7874, 1)
pld2Pattern = (pld2Body, pld2EdgeBody, pld2Pin, 0.7874, 2)

#Modified BH models
bhRegions = [(((0.7, 3.0, 4.0), (-2.5, -3.0, -0.5)), 1), (((6.5, 3.0, 4.0), (4.5, -3.0, -0.5)), 2)]
bhAttributedBody = model.AttributedMesh(name="PatBHAttributed", regions=bhRegions)
bhAttributedBody.append(bhBody)
bhAttributedBody.visualAppearance = bhBody.appearance()
bhPattern = (bhAttributedBody, bhPin, 1.0)

def createModel(pattern, count, name):
    if pattern[4] < 1 or pattern[4] > 2:
        raise Exception()
    if count / pattern[4] < 2:
        raise Exception()

    body = model.Mesh(name="%s_%uBody" % (name, count))
    body.visualAppearance = pattern[0].appearance()
    pins = []

    shift = pattern[3] / 2. if pattern[4] > 1 else 0.
    columns = count / pattern[4]
    for i in range(0, columns):
        if i == 0:
            segment = copy.deepcopy(pattern[1])
            segment.rotate([0., 0., 1.], math.pi)
        elif i == columns - 1:
            segment = copy.deepcopy(pattern[1])
            segment.rotate([0., 0., 1.], 0.)
        else:
            segment = copy.deepcopy(pattern[0])
        segment.translate([float(i) * pattern[3], shift, 0.])
        body.append(segment)

        pin = model.Mesh(parent=pattern[2], name="%s_%uPin%u" % (name, count, (i + 1)))
        pin.translate([float(i) * pattern[3], shift, 0.001])
        pins.append(pin)
    body.translate([0., 0., 0.001])
    body.optimize()
    return [body] + pins

def createBHBody(pattern, width, count, name):
    DEFAULT_WIDTH = 20.34
    delta = (width - DEFAULT_WIDTH) / 2. / 2.54
    shift = pattern[2] / 2.

    leftPart, rightPart = model.Transform(), model.Transform()
    leftPart.translate([-delta, 0., 0.])
    rightPart.translate([delta, 0., 0.])
    transforms = [model.Transform(), leftPart, rightPart]
    body = copy.deepcopy(pattern[0])
    body.applyTransforms(transforms)
    body.translate([delta, shift, 0.001])

    pins = []

    for i in range(0, count / 2):
        pin = model.Mesh(parent=pattern[1], name="%s_%uPin%u" % (name, count, (i + 1)))
        pin.translate([float(i) * pattern[2], shift, 0.001])
        pins.append(pin)

    return [body] + pins

models = []

models.append((createBHBody(bhPattern, 20.34, 10, "BH"), "bh-10"))
models.append((createBHBody(bhPattern, 25.48, 14, "BH"), "bh-14"))
models.append((createBHBody(bhPattern, 27.90, 16, "BH"), "bh-16"))
models.append((createBHBody(bhPattern, 33.05, 20, "BH"), "bh-20"))
models.append((createBHBody(bhPattern, 38.08, 24, "BH"), "bh-24"))
models.append((createBHBody(bhPattern, 40.68, 26, "BH"), "bh-26"))
models.append((createBHBody(bhPattern, 45.70, 30, "BH"), "bh-30"))
models.append((createBHBody(bhPattern, 50.80, 34, "BH"), "bh-34"))
models.append((createBHBody(bhPattern, 58.46, 40, "BH"), "bh-40"))

models.append((createModel(pldPattern, 6, "PLD"), "pld-6"))
models.append((createModel(pldPattern, 8, "PLD"), "pld-8"))
models.append((createModel(pldPattern, 10, "PLD"), "pld-10"))
models.append((createModel(pldPattern, 12, "PLD"), "pld-12"))
models.append((createModel(pldPattern, 14, "PLD"), "pld-14"))
models.append((createModel(pldPattern, 16, "PLD"), "pld-16"))
models.append((createModel(pldPattern, 18, "PLD"), "pld-18"))
models.append((createModel(pldPattern, 20, "PLD"), "pld-20"))

models.append((createModel(plsPattern, 2, "PLS"), "pls-2"))
models.append((createModel(plsPattern, 3, "PLS"), "pls-3"))
models.append((createModel(plsPattern, 4, "PLS"), "pls-4"))
models.append((createModel(plsPattern, 5, "PLS"), "pls-5"))
models.append((createModel(plsPattern, 6, "PLS"), "pls-6"))
models.append((createModel(plsPattern, 8, "PLS"), "pls-8"))
models.append((createModel(plsPattern, 10, "PLS"), "pls-10"))

models.append((createModel(pld2Pattern, 6, "PLD2"), "pld2-6"))
models.append((createModel(pld2Pattern, 8, "PLD2"), "pld2-8"))
models.append((createModel(pld2Pattern, 10, "PLD2"), "pld2-10"))
models.append((createModel(pld2Pattern, 12, "PLD2"), "pld2-12"))
models.append((createModel(pld2Pattern, 14, "PLD2"), "pld2-14"))
models.append((createModel(pld2Pattern, 16, "PLD2"), "pld2-16"))
models.append((createModel(pld2Pattern, 18, "PLD2"), "pld2-18"))
models.append((createModel(pld2Pattern, 20, "PLD2"), "pld2-20"))

models.append((createModel(pls2Pattern, 2, "PLS2"), "pls2-2"))
models.append((createModel(pls2Pattern, 3, "PLS2"), "pls2-3"))
models.append((createModel(pls2Pattern, 4, "PLS2"), "pls2-4"))
models.append((createModel(pls2Pattern, 5, "PLS2"), "pls2-5"))
models.append((createModel(pls2Pattern, 6, "PLS2"), "pls2-6"))
models.append((createModel(pls2Pattern, 8, "PLS2"), "pls2-8"))
models.append((createModel(pls2Pattern, 10, "PLS2"), "pls2-10"))

#Jumpers
plsJumperBody = copy.deepcopy(plsJumper)
plsJumperBody.translate([0., 0., 0.001])

plsJumper2Pin = [plsJumperBody] + createModel(plsPattern, 2, "MJ")
models.append((plsJumper2Pin, "mj-2"))
plsJumper3Pin = [plsJumperBody] + createModel(plsPattern, 3, "MJ")
models.append((plsJumper3Pin, "mj-3"))

pls2JumperBody = copy.deepcopy(pls2Jumper)
pls2JumperBody.translate([0., 0., 0.001])

pls2Jumper2Pin = [pls2JumperBody] + createModel(pls2Pattern, 2, "MJ2")
models.append((pls2Jumper2Pin, "mj2-2"))
pls2Jumper3Pin = [pls2JumperBody] + createModel(pls2Pattern, 3, "MJ2")
models.append((pls2Jumper3Pin, "mj2-3"))

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

#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# generator_caps.py
# Copyright (C) 2016 xent
# Project is distributed under the terms of the GNU General Public License v3.0

import argparse
import copy
import json

import curves
import helpers
import model
import render_ogl41
import vrml_export
import vrml_export_kicad
import vrml_import
import x3d_export
import x3d_import


class MeshOptions:
    def __init__(self, smooth, normals):
        self.smooth = smooth
        self.normals = normals


parser = argparse.ArgumentParser()
parser.add_argument("-d", dest="debug", help="show debug information", default=False, action="store_true")
parser.add_argument("-f", dest="format", help="output file format", default="x3d")
parser.add_argument("-i", dest="input", help="input file with part descriptions", default="")
parser.add_argument("-m", dest="materials", help="file with materials", default="")
parser.add_argument("-o", dest="output", help="write models to specified directory", default="")
parser.add_argument("-v", dest="view", help="render models", default=False, action="store_true")
parser.add_argument("--fast", dest="fast", help="disable visual effects", default=False, action="store_true")
parser.add_argument("--normals", dest="normals", help="show normals", default=False, action="store_true")
parser.add_argument("--smooth", dest="smooth", help="use smooth shading", default=False, action="store_true")
options = parser.parse_args()

meshOptions = MeshOptions(options.smooth, options.normals)

def debug(text):
    global options

    if options.debug:
        print(text)

if options.debug:
    render_ogl41.debugEnabled = True
    vrml_export.debugEnabled = True
    vrml_export_kicad.debugEnabled = True
    vrml_import.debugEnabled = True
    x3d_import.debugEnabled = True
    x3d_export.debugEnabled = True

def buildCapacitorCurve(radius, height, curvature, bandOffset, capRadius, capDepth, chamfer=None,
        edgeDetails=3, bandDetails=4):
    if capRadius is not None and capDepth is not None and chamfer is None:
        raise Exception()

    curve = []

    #Bottom cap
    if capRadius is not None:
        if capDepth is not None:
            curve.append(curves.Line((capRadius, 0., capDepth - chamfer), (capRadius, 0., chamfer), 1))
            curve.append(curves.Line((capRadius, 0., chamfer), (capRadius + chamfer, 0., 0.), 1))
            curve.append(curves.Line((capRadius + chamfer, 0., 0.), (radius - curvature, 0., 0.), 1))
        else:
            curve.append(curves.Line((capRadius, 0., 0.), (radius - curvature, 0., 0.), 1))

    #Plastic
    curve.append(curves.Bezier((radius - curvature, 0., 0.), (curvature / 2., 0., 0.),
            (radius, 0., curvature), (0., 0., -curvature / 2.), edgeDetails))
    curve.append(curves.Line((radius, 0., curvature), (radius, 0., bandOffset - curvature * 2.), 1))
    curve.append(curves.Bezier((radius, 0., bandOffset - curvature * 2.), (0., 0., curvature),
            (radius - curvature, 0., bandOffset), (0., 0., -curvature), bandDetails))
    curve.append(curves.Bezier((radius - curvature, 0., bandOffset), (0., 0., curvature),
            (radius, 0., bandOffset + curvature * 2.), (0., 0., -curvature), bandDetails))
    curve.append(curves.Line((radius, 0., bandOffset + curvature * 2.), (radius, 0., height - curvature), 1))
    curve.append(curves.Bezier((radius, 0., height - curvature), (0., 0., curvature / 2.),
            (radius - curvature, 0., height), (curvature / 2., 0., 0.), edgeDetails))

    #Top cap
    if capRadius is not None:
        if capDepth is not None:
            curve.append(curves.Line((radius - curvature, 0., height), (capRadius + chamfer, 0., height), 1))
            curve.append(curves.Line((capRadius + chamfer, 0., height), (capRadius, 0., height - chamfer), 1))
            curve.append(curves.Line((capRadius, 0., height - chamfer), (capRadius, 0., height - capDepth), 1))
        else:
            curve.append(curves.Line((radius - curvature, 0., height), (capRadius, 0., height), 1))

    return curve

def buildPinCurve(radius, height, curvature, edgeDetails=2):
    curve = []

    curve.append(curves.Bezier((radius - curvature, 0., -height), (curvature / 2., 0., 0.),
            (radius, 0., -height + curvature), (0., 0., -curvature / 2.), edgeDetails))
    curve.append(curves.Line((radius, 0., curvature - height), (radius, 0., 0.), 1))

    return curve

def buildCapacitorBody(curve, edges, polarized, materials, name):
    global meshOptions

    slices = curves.rotate(curve, (0., 0., 1.), edges)
    meshes = []

    bottomCap = curves.createTriCapMesh(slices, True)
    bottomCap.visualAppearance.material = materials["Bottom"]
    bottomCap.visualAppearance.normals = meshOptions.normals
    bottomCap.visualAppearance.smooth = meshOptions.smooth
    bottomCap.ident = name + "BottomCap"
    meshes.append(bottomCap)

    topCap = curves.createTriCapMesh(slices, False)
    topCap.visualAppearance.material = materials["Top"]
    topCap.visualAppearance.normals = meshOptions.normals
    topCap.visualAppearance.smooth = meshOptions.smooth
    topCap.ident = name + "TopCap"
    meshes.append(topCap)

    if polarized:
        body = curves.createRotationMesh(slices[1:], False)
        body.visualAppearance.material = materials["Body"]
        body.visualAppearance.normals = meshOptions.normals
        body.visualAppearance.smooth = meshOptions.smooth
        body.ident = name + "Body"
        meshes.append(body)

        mark = curves.createRotationMesh([slices[-1]] + slices[0:2], False)
        mark.visualAppearance.material = materials["Mark"]
        mark.visualAppearance.normals = meshOptions.normals
        mark.visualAppearance.smooth = meshOptions.smooth
        mark.ident = name + "Mark"
        meshes.append(mark)
    else:
        body = curves.createRotationMesh(slices, True)
        body.visualAppearance.material = materials["Body"]
        body.visualAppearance.normals = meshOptions.normals
        body.visualAppearance.smooth = meshOptions.smooth
        body.ident = name + "Body"
        meshes.append(body)

    return meshes

def buildCapacitorPin(curve, edges):
    global meshOptions

    slices = curves.rotate(curve, (0., 0., 1.), edges)

    pin = curves.createRotationMesh(slices, True)
    pin.append(curves.createTriCapMesh(slices, True))
    pin.optimize()

    pin.visualAppearance.normals = meshOptions.normals
    pin.visualAppearance.smooth = meshOptions.smooth

    return pin

def buildCapacitor(title, materials, polarized, radius, height, curvature, edges, band, capRadius, capDepth, capChamfer,
            pinRadius, pinHeight, pinCurvature, pinEdges, pinSpacing):
    meshes = []
    bodyCurve = buildCapacitorCurve(radius, height, curvature, band, capRadius, capDepth, capChamfer)
    meshes.extend(buildCapacitorBody(bodyCurve, edges, polarized, materials, title))

    pinCurve = buildPinCurve(pinRadius, pinHeight, pinCurvature)
    pinMesh = buildCapacitorPin(pinCurve, pinEdges)
    pinMesh.visualAppearance.material = materials["Pin"]
    pinMesh.ident = title + "Pin"

    posPin = copy.deepcopy(pinMesh)
    posPin.translate([-pinSpacing / 2., 0., 0.])
    posPin.ident = pinMesh.ident + "Pos"
    meshes.append(posPin)
    negPin = copy.deepcopy(pinMesh)
    negPin.translate([pinSpacing / 2., 0., 0.])
    negPin.ident = pinMesh.ident + "Neg"
    meshes.append(negPin)

    return meshes

def demangle(title):
    return title.replace("C-", "Cap").replace("CP-", "Cap").replace("R-", "Radial").replace("A-", "Axial")

def metricToImperial(value):
    return value / 2.54

materials = {}
models = []

if options.materials != "":
    materials = helpers.loadMaterials(options.materials)

for matGroup in ["Body", "Mark", "Top", "Bottom", "Pin"]:
    if matGroup not in materials.keys():
        materials[matGroup] = model.Material()
        materials[matGroup].color.ident = matGroup

if options.input != "":
    content = json.loads(open(options.input, "rb").read())
    if "parts" in content.keys():
        for part in content["parts"]:
            cap = buildCapacitor(
                    title=demangle(part["title"]),
                    materials=materials,
                    polarized=part["polarized"],
                    radius=metricToImperial(part["body"]["radius"]),
                    height=metricToImperial(part["body"]["height"]),
                    curvature=metricToImperial(part["body"]["curvature"]),
                    edges=part["body"]["edges"],
                    band=metricToImperial(part["body"]["band"]),
                    capRadius=metricToImperial(part["caps"]["radius"]),
                    capDepth=metricToImperial(part["caps"]["depth"]),
                    capChamfer=metricToImperial(part["caps"]["chamfer"]),
                    pinRadius=metricToImperial(part["pins"]["radius"]),
                    pinHeight=metricToImperial(part["pins"]["height"]),
                    pinCurvature=metricToImperial(part["pins"]["curvature"]),
                    pinEdges=part["pins"]["edges"],
                    pinSpacing=metricToImperial(part["pins"]["spacing"]))
            models.append((cap, part["title"].lower()))

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

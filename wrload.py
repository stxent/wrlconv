#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# wrload.py
# Copyright (C) 2011 xent
# Project is distributed under the terms of the GNU General Public License v3.0

import argparse
import math
import os
import re

import render_ogl41
import vrml_import
import vrml_export
import x3d_import
import x3d_export

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
parser.add_argument("-x", dest="simplified", help="export with simplified syntax", default=False, action="store_true")
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

if options.pattern != "":
    exportList = filter(lambda mesh: re.search(options.pattern, mesh.ident, re.S) is not None, exportList)

if options.output != "":
    extension = os.path.splitext(options.output)[1][1:].lower()
    if extension == "wrl":
        spec = vrml_export.VRML_KICAD if options.simplified else vrml_export.VRML_STRICT
        vrml_export.store(exportList, options.output, spec)
    elif extension == "x3d":
        x3d_export.store(exportList, options.output)

if options.view:
    render = render_ogl41.Render(exportList)

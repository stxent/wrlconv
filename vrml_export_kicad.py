#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# vrml_export.py
# Copyright (C) 2013 xent
# Project is distributed under the terms of the GNU General Public License v3.0

import numpy
import time

try:
    import model
except ImportError:
    from . import model

debugEnabled = False

def debug(text):
    if debugEnabled:
        print(text)

def store(data, path):
    exportedMaterials = []

    def encodeAppearance(material, level):
        def calcIntensity(ambient, diffuse):
            result = 0.
            for index in range(0, 3):
                if diffuse[index]:
                    result += ambient[index] / diffuse[index]
            return result / 3.

        output = ""
        ambIntensity = min(calcIntensity(material.color.ambient, material.color.diffuse), 1.0)
        output += "%sappearance Appearance {\n" % ("\t" * level)

        if material in exportedMaterials:
            exported = exportedMaterials[exportedMaterials.index(material)]
            output += "%smaterial USE MA_%s\n" % ("\t" * (level + 1), exported.color.ident)
            debug("Export: reused material %s instead of %s" % (exported.color.ident, material.color.ident))
        else:
            output += "%smaterial DEF MA_%s Material {\n" % ("\t" * (level + 1), material.color.ident)
            output += "%sdiffuseColor %f %f %f\n" % tuple(["\t" * (level + 2)] + material.color.diffuse.tolist())
            output += "%sambientIntensity %f\n" % ("\t" * (level + 2), ambIntensity)
            output += "%sspecularColor %f %f %f\n" % tuple(["\t" * (level + 2)] + material.color.specular.tolist())
            output += "%semissiveColor %f %f %f\n" % tuple(["\t" * (level + 2)] + material.color.emissive.tolist())
            output += "%sshininess %f\n" % ("\t" * (level + 2), material.color.shininess)
            output += "%stransparency %f\n" % ("\t" * (level + 2), material.color.transparency)
            output += "%s}\n" % ("\t" * (level + 1))
            exportedMaterials.append(material)

        output += "%s}\n" % ("\t" * level)
        return output

    def encodeGeometry(mesh, transform, level):
        output = ""
        output += "%sgeometry IndexedFaceSet {\n" % ("\t" * level)
        output += "%ssolid FALSE\n" % ("\t" * (level + 1))

        geoVertices, geoPolygons = mesh.geometry()

        output += "%scoord DEF FS_%s Coordinate {\n" % ("\t" * (level + 1), mesh.ident)
        output += "%spoint [\n" % ("\t" * (level + 2))
        for vert in geoVertices:
            resultingVert = transform.process(vert)
            output += "\t"
            output += " ".join(map(lambda x: str(round(x, 6)), resultingVert))
            output += "\n"
        output += "%s]\n" % ("\t" * (level + 2))
        output += "%s}\n" % ("\t" * (level + 1))

        output += "%scoordIndex [\n" % ("\t" * (level + 1))
        for poly in map(lambda poly: poly + [-1], geoPolygons):
            output += "\t"
            output += " ".join(map(str, poly))
            output += "\n"
        output += "%s]\n" % ("\t" * (level + 1))

        output += "%s}\n" % ("\t" * level)
        return output

    def encodeShape(mesh, transform, level):
        output = ""
        output += "%sShape {\n" % ("\t" * level)

        output += encodeAppearance( mesh.appearance().material, level + 1)
        output += encodeGeometry(mesh, transform, level + 1)

        output += "%s}\n" % ("\t" * level)
        return output

    def encodeGroup(mesh, topTransform, topName, level):
        output = ''
        transform = topTransform if mesh.transform is None else mesh.transform

        output += "%sDEF ME_%s_%s Group {\n" % ("\t" * level, topName, mesh.ident)
        output += "%schildren [\n" % ("\t" * (level + 1))
        output += encodeShape(mesh, transform, level + 2)
        output += "%s]\n" % ("\t" * (level + 1))
        output += "%s}\n" % ("\t" * level)
        return output

    def encodeTransform(mesh, level=0):
        output = ''
        started = time.time()

        output += "%sDEF OB_%s Transform {\n" % ("\t" * level, mesh.ident)
        output += "%stranslation 0.0 0.0 0.0\n" % ("\t" * (level + 1))
        output += "%srotation 1.0 0.0 0.0 0.0\n" % ("\t" * (level + 1))
        output += "%sscale 1.0 1.0 1.0\n" % ("\t" * (level + 1))
        output += "%schildren [\n" % ("\t" * (level + 1))

        parent = mesh if mesh.parent is None else mesh.parent
        if mesh.transform is not None:
            transform = mesh.transform
        else:
            transform = model.Transform()
        output += encodeGroup(parent, transform, mesh.ident, level + 2)

        output += "%s]\n" % ("\t" * (level + 1))
        output += "%s}\n" % ("\t" * level)

        debug('Mesh exported in %f, name %s' % (time.time() - started, mesh.ident))
        return output

    sortedData = [entry for entry in data if entry.appearance().material.color.transparency <= 0.001]
    sortedData += [entry for entry in data if entry.appearance().material.color.transparency > 0.001]

    out = open(path, 'wb')
    out.write('#VRML V2.0 utf8\n#Created by vrml_export_kicad.py\n'.encode('utf-8'))
    for shape in sortedData:
        out.write(encodeTransform(shape).encode('utf-8'))
    out.close()

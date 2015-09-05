#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# vrml_export.py
# Copyright (C) 2013 xent
# Project is distributed under the terms of the GNU General Public License v3.0

import numpy
import time

import model

debugEnabled = False

def debug(text):
    if debugEnabled:
        print(text)

def store(data, path):
    exportedMaterials = []

    def writeAppearance(stream, material, level):
        def calcIntensity(ambient, diffuse):
            result = 0.
            for index in range(0, 3):
                if diffuse[index]:
                    result += ambient[index] / diffuse[index]
            return result / 3.

        ambIntensity = calcIntensity(material.color.ambient, material.color.diffuse)
        if ambIntensity > 1.:
            ambIntensity = 1.
        stream.write("%sappearance Appearance {\n" % ("\t" * level))

        if material in exportedMaterials:
            stream.write("%smaterial USE MA_%s\n" % ("\t" * (level + 1), material.color.ident))
            debug("Export: reused material %s" % material.color.ident)
        else:
            stream.write("%smaterial DEF MA_%s Material {\n" % ("\t" * (level + 1), material.color.ident))
            stream.write("%sdiffuseColor %f %f %f\n" % tuple(["\t" * (level + 2)] + material.color.diffuse.tolist()))
            stream.write("%sambientIntensity %f\n" % ("\t" * (level + 2), ambIntensity))
            stream.write("%sspecularColor %f %f %f\n" % tuple(["\t" * (level + 2)] + material.color.specular.tolist()))
            stream.write("%semissiveColor %f %f %f\n" % tuple(["\t" * (level + 2)] + material.color.emissive.tolist()))
            stream.write("%sshininess %f\n" % ("\t" * (level + 2), material.color.shininess))
            stream.write("%stransparency %f\n" % ("\t" * (level + 2), material.color.transparency))
            stream.write("%s}\n" % ("\t" * (level + 1)))
            exportedMaterials.append(material)

        stream.write("%s}\n" % ("\t" * level))

    def writeGeometry(stream, mesh, transform, level):
        stream.write("%sgeometry IndexedFaceSet {\n" % ("\t" * level))
        stream.write("%ssolid FALSE\n" % ("\t" * (level + 1)))

        geoVertices, geoPolygons = mesh.geometry()

        stream.write("%scoord DEF FS_%s Coordinate {\n" % ("\t" * (level + 1), mesh.ident))
        stream.write("%spoint [\n" % ("\t" * (level + 2)))
        for vert in geoVertices:
            resultingVert = transform.process(vert)
            stream.write("\t")
            stream.write(" ".join(map(lambda x: str(round(x, 6)), resultingVert)))
            stream.write("\n")
        stream.write("%s]\n" % ("\t" * (level + 2)))
        stream.write("%s}\n" % ("\t" * (level + 1)))

        stream.write("%scoordIndex [\n" % ("\t" * (level + 1)))
        for poly in map(lambda poly: poly + [-1], geoPolygons):
            stream.write("\t")
            stream.write(" ".join(map(str, poly)))
            stream.write("\n")
        stream.write("%s]\n" % ("\t" * (level + 1)))

        stream.write("%s}\n" % ("\t" * level))

    def writeShape(stream, mesh, transform, level):
        stream.write("%sShape {\n" % ("\t" * level))

        writeAppearance(stream, mesh.appearance().material, level + 1)
        writeGeometry(stream, mesh, transform, level + 1)

        stream.write("%s}\n" % ("\t" * level))

    def writeGroup(stream, mesh, topTransform, topName, level):
        transform = topTransform if mesh.transform is None else mesh.transform

        stream.write("%sDEF ME_%s_%s Group {\n" % ("\t" * level, topName, mesh.ident))
        stream.write("%schildren [\n" % ("\t" * (level + 1)))
        writeShape(stream, mesh, transform, level + 2)
        stream.write("%s]\n" % ("\t" * (level + 1)))
        stream.write("%s}\n" % ("\t" * level))

    def writeTransform(stream, mesh, level=0):
        started = time.time()

        stream.write("%sDEF OB_%s Transform {\n" % ("\t" * level, mesh.ident))
        stream.write("%stranslation 0.0 0.0 0.0\n" % ("\t" * (level + 1)))
        stream.write("%srotation 1.0 0.0 0.0 0.0\n" % ("\t" * (level + 1)))
        stream.write("%sscale 1.0 1.0 1.0\n" % ("\t" * (level + 1)))
        stream.write("%schildren [\n" % ("\t" * (level + 1)))

        parent = mesh if mesh.parent is None else mesh.parent
        if mesh.transform is not None:
            transform = mesh.transform
        else:
            transform = model.Transform()
        writeGroup(stream, parent, transform, mesh.ident, level + 2)

        stream.write("%s]\n" % ("\t" * (level + 1)))
        stream.write("%s}\n" % ("\t" * level))

        debug("Mesh exported in %f, name %s" % (time.time() - started, mesh.ident))

    sortedData = filter(lambda entry: entry.appearance().material.color.transparency <= 0.001, data)
    sortedData += filter(lambda entry: entry.appearance().material.color.transparency > 0.001, data)

    out = open(path, "wb")
    out.write("#VRML V2.0 utf8\n#Created by vrml_export_kicad.py\n")
    for shape in sortedData:
        writeTransform(out, shape)
    out.close()

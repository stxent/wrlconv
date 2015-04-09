#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# vrml_export.py
# Copyright (C) 2013 xent
# Project is distributed under the terms of the GNU General Public License v3.0

import numpy
import time

import model

VRML_STRICT, VRML_KICAD, VRML_EXT = range(0, 3)

debugEnabled = False

def debug(text):
    if debugEnabled:
        print(text)

def store(data, path, spec=VRML_STRICT):
    exportedGroups, exportedMaterials = [], []

    def writeAppearance(spec, stream, material, level):
        ambIntensity = sum(map(lambda i: material.color.ambient[i] / material.color.diffuse[i] / 3., range(0, 3)))
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

            def writeTexture(stream, path, name, level):
                if name != "":
                    stream.write("%stexture DEF %s ImageTexture {\n" % ("\t" * level, name))
                else:
                    stream.write("%stexture %s {\n" % ("\t" * level))
                stream.write("%surl \"%s\"\n" % ("\t" * (level + 1), path))
                stream.write("%s}\n" % ("\t" * level))

            if material.diffuse is not None:
                writeTexture(stream, material.diffuse.path[0], material.diffuse.ident, level + 1)
            if spec == VRML_EXT and material.normalmap is not None:
                writeTexture(stream, material.normalmap.path[0], material.normalmap.ident, level + 1)
            if spec == VRML_EXT and material.specular is not None:
                writeTexture(stream, material.specular.path[0], material.specular.ident, level + 1)

        stream.write("%s}\n" % ("\t" * level))

    def writeGeometry(spec, stream, mesh, transform, level):
        stream.write("%sgeometry IndexedFaceSet {\n" % ("\t" * level))

        appearance = mesh.appearance()
        stream.write("%ssolid %s\n" % ("\t" * (level + 1), "TRUE" if appearance.solid else "FALSE"))
        stream.write("%ssmooth %s\n" % ("\t" * (level + 1), "TRUE" if appearance.smooth else "FALSE"))

        geoVertices, geoPolygons = mesh.geometry()

        stream.write("%scoord DEF FS_%s Coordinate {\n" % ("\t" * (level + 1), mesh.ident))
        stream.write("%spoint [\n" % ("\t" * (level + 2)))
        stream.write("\t" * (level + 3))
        vertices = []
        for vert in geoVertices:
            vertices.extend(vert if transform is None else transform.process(vert))
        stream.write(" ".join(map(str, vertices)))
        stream.write("\n")
        stream.write("%s]\n" % ("\t" * (level + 2)))
        stream.write("%s}\n" % ("\t" * (level + 1)))

        stream.write("%scoordIndex [\n" % ("\t" * (level + 1)))
        stream.write("\t" * (level + 2))
        indices = []
        [indices.extend(poly) for poly in map(lambda poly: poly + [-1], geoPolygons)]
        stream.write(" ".join(map(str, indices)))
        stream.write("\n")
        stream.write("%s]\n" % ("\t" * (level + 1)))

        material = appearance.material
        if any(texture is not None for texture in [material.diffuse, material.normalmap, material.specular]):
            texVertices, texPolygons = mesh.texture()

            stream.write("%stexCoord TextureCoordinate {\n" % ("\t" * (level + 1)))
            stream.write("%spoint [\n" % ("\t" * (level + 2)))
            stream.write("\t" * (level + 3))
            vertices = []
            [vertices.extend(vertex) for vertex in texVertices]
            stream.write(" ".join(map(str, vertices)))
            stream.write("\n")
            stream.write("%s]\n" % ("\t" * (level + 2)))
            stream.write("%s}\n" % ("\t" * (level + 1)))

            stream.write("%stexCoordIndex [\n" % ("\t" * (level + 1)))
            stream.write("\t" * (level + 2))
            indices = []
            [indices.extend(poly) for poly in map(lambda poly: poly + [-1], texPolygons)]
            stream.write(" ".join(map(str, indices)))
            stream.write("\n")
            stream.write("%s]\n" % ("\t" * (level + 1)))

        stream.write("%s}\n" % ("\t" * level))

    def writeShape(spec, stream, mesh, transform, level):
        stream.write("%sShape {\n" % ("\t" * level))

        writeAppearance(spec, stream, mesh.appearance().material, level + 1)
        writeGeometry(spec, stream, mesh, transform, level + 1)

        stream.write("%s}\n" % ("\t" * level))

    def writeGroup(spec, stream, mesh, level):
        alreadyExported = filter(lambda group: group.ident == mesh.ident, exportedGroups)

        if len(alreadyExported) == 0:
            stream.write("%sDEF ME_%s Group {\n" % ("\t" * level, mesh.ident))
            stream.write("%schildren [\n" % ("\t" * (level + 1)))
            writeShape(spec, stream, mesh, mesh.transform, level + 2)
            stream.write("%s]\n" % ("\t" * (level + 1)))
            stream.write("%s}\n" % ("\t" * level))
            exportedGroups.append(mesh)
        else:
            stream.write("%sUSE ME_%s\n" % ("\t" * level, mesh.ident))
            debug("Export: reused group %s" % mesh.ident)

    def writeTransform(spec, stream, mesh, level=0):
        started = time.time()

        if mesh.transform is None:
            translation = [0., 0., 0.]
        else:
            translation = numpy.array(mesh.transform.value)[:,3][0:3].tolist()

        stream.write("%sDEF OB_%s Transform {\n" % ("\t" * level, mesh.ident))
        stream.write("%stranslation %f %f %f\n" % tuple(["\t" * (level + 1)] + translation))
        stream.write("%srotation 1.0 0.0 0.0 0.0\n" % ("\t" * (level + 1)))
        stream.write("%sscale 1.0 1.0 1.0\n" % ("\t" * (level + 1)))
        stream.write("%schildren [\n" % ("\t" * (level + 1)))

        parent = mesh if mesh.parent is None else mesh.parent
        writeGroup(spec, stream, parent, level + 2)

        stream.write("%s]\n" % ("\t" * (level + 1)))
        stream.write("%s}\n" % ("\t" * level))

        debug("Mesh exported in %f, name %s" % (time.time() - started, mesh.ident))

    out = open(path, "wb")
    out.write("#VRML V2.0 utf8\n#Created by vrml_export.py\n")
    for shape in data:
        writeTransform(spec, out, shape)
    out.close()

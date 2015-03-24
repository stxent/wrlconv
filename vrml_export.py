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

def exportVrml(spec, path, data):
    exportedGroups, exportedMaterials = [], []

    def writeAppearance(spec, stream, material, level):
        ambIntensity = sum(map(lambda i: material.color.ambient[i] / material.color.diffuse[i] / 3., range(0, 3)))
        stream.write("%sappearance Appearance {\n" % ("\t" * level))

        if material in exportedMaterials:
            stream.write("%smaterial USE MA_%s\n" % ("\t" * (level + 1), material.color.ident))
            debug("Export: reused material %s" % material.color.ident)
        else:
            stream.write("%smaterial DEF MA_%s Material {\n" % ("\t" * (level + 1), material.color.ident))
            stream.write("%sdiffuseColor %f %f %f\n" % ("\t" * (level + 2), material.color.diffuse[0],\
                    material.color.diffuse[1], material.color.diffuse[2]))
            stream.write("%sambientIntensity %f\n" % ("\t" * (level + 2), ambIntensity))
            stream.write("%sspecularColor %f %f %f\n" % ("\t" * (level + 2), material.color.specular[0],\
                    material.color.specular[1], material.color.specular[2]))
            stream.write("%semissiveColor %f %f %f\n" % ("\t" * (level + 2), material.color.emissive[0],\
                    material.color.emissive[1], material.color.emissive[2]))
            stream.write("%sshininess %f\n" % ("\t" * (level + 2), material.color.shininess))
            stream.write("%stransparency %f\n" % ("\t" * (level + 2), material.color.transparency))
            stream.write("%s}\n" % ("\t" * (level + 1)))
            exportedMaterials.append(material)

        if spec != VRML_KICAD:
            #FIXME Print relative path
            if material.diffuse is not None:
                stream.write("%stexture DEF %s ImageTexture {\n"\
                        % ("\t" * (level + 1), material.diffuse.ident))
                stream.write("%surl \"%s\"\n" % ("\t" * (level + 2), material.diffuse.path))
                stream.write("%s}\n" % ("\t" * (level + 1)))
            if spec == VRML_EXT:
                if material.normalmap is not None:
                    stream.write("%stexture DEF normalmap_%s ImageTexture {\n"\
                            % ("\t" * (level + 1), material.normalmap.ident))
                    stream.write("%surl \"%s\"\n" % ("\t" * (level + 2), material.normalmap.path))
                    stream.write("%s}\n" % ("\t" * (level + 1)))
                if material.specular is not None:
                    stream.write("%stexture DEF specular_%s ImageTexture {\n"\
                            % ("\t" * (level + 1), material.specular.ident))
                    stream.write("%surl \"%s\"\n" % ("\t" * (level + 2), material.specular.path))
                    stream.write("%s}\n" % ("\t" * (level + 1)))

        stream.write("%s}\n" % ("\t" * level))

    def writeGeometry(spec, stream, mesh, level):
        stream.write("%sgeometry IndexedFaceSet {\n" % ("\t" * level))

        appearance = mesh.appearance()
        if spec == VRML_KICAD:
            stream.write("%ssolid FALSE\n" % ("\t" * (level + 1)))
        else:
            stream.write("%ssolid %s\n" % ("\t" * (level + 1), "TRUE" if appearance["solid"] else "FALSE"))
            stream.write("%ssmooth %s\n" % ("\t" * (level + 1), "TRUE" if appearance["smooth"] else "FALSE"))

        geoVertices, geoPolygons = mesh.geometry()

        stream.write("%scoord DEF FS_%s Coordinate {\n" % ("\t" * (level + 1), mesh.ident))
        stream.write("%spoint [\n" % ("\t" * (level + 2)))
        stream.write("\t" * (level + 3))
        vertices = []
        for srcVert in geoVertices:
            vert = numpy.matrix([[srcVert[0]], [srcVert[1]], [srcVert[2]], [1.]])
            if mesh.transform is not None:
                vert = mesh.transform.value * vert
            vertices.extend([vert[0, 0], vert[1, 0], vert[2, 0]])
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

        material = appearance["material"]
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

    def writeShape(spec, stream, mesh, level):
        stream.write("%sShape {\n" % ("\t" * level))

        writeAppearance(spec, stream, mesh.appearance()["material"], level + 1)
        writeGeometry(spec, stream, mesh, level + 1)

        stream.write("%s}\n" % ("\t" * level))

    def writeGroup(spec, stream, mesh, level):
        if spec != VRML_KICAD:
            alreadyExported = filter(lambda group: group.ident == mesh.ident, exportedGroups)
        else:
            alreadyExported = []

        if spec == VRML_KICAD or len(alreadyExported) == 0:
            stream.write("%sDEF ME_%s Group {\n" % ("\t" * level, mesh.ident))
            stream.write("%schildren [\n" % ("\t" * (level + 1)))
            writeShape(spec, stream, mesh, level + 2)
            stream.write("%s]\n" % ("\t" * (level + 1)))
            stream.write("%s}\n" % ("\t" * level))
            if spec != VRML_KICAD:
                exportedGroups.append(mesh)
        else:
            stream.write("%sUSE ME_%s\n" % ("\t" * level, mesh.ident))
            debug("Export: reused group %s" % mesh.ident)

    def writeTransform(spec, stream, mesh, level=0):
        started = time.time()

        if mesh.transform is None:
            translation = [0., 0., 0.]
        else:
            column = mesh.transform.value[:,3][0:3]
            translation = [column[0], column[1], column[2]]
        stream.write("%sDEF OB_%s Transform {\n" % ("\t" * level, mesh.ident))
        stream.write("%stranslation %f %f %f\n" % ("\t" * (level + 1), translation[0], translation[1], translation[2]))
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

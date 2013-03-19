#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# vrml_export.py
# Copyright (C) 2013 xent
# Project is distributed under the terms of the GNU General Public License v3.0

import numpy
import math

import model

VRML_STRICT, VRML_KICAD, VRML_EXT = range(0, 3)

def exportVRML(spec, path, data):
    def writeMaterial(spec, stream, app):
        #ambIntensity = sum[map(lambda i: app.color.diffuse[i] / app.color.ambient[i] / 3, range(0, 3))]
        ambIntensity = 0.2
        stream.write("appearance Appearance {\n"
                     "material DEF MAT_%s Material {\n" % app.color.ident +
                     "diffuseColor %f %f %f\n" % tuple(app.color.diffuse) +
                     "ambientIntensity %f\n" % ambIntensity +
                     "specularColor %f %f %f\n" % tuple(app.color.specular) +
                     "emissiveColor %f %f %f\n" % tuple(app.color.emissive) +
                     "shininess %f\n" % app.color.shininess +
                     "transparency %f\n" % app.color.transparency +
                     "}\n")
        if spec != VRML_KICAD:
            if app.diffuse is not None:
                stream.write("texture DEF diffuse_%s ImageTexture {\n" % app.diffuse.ident +
                             "url \"%s\"\n" % app.diffuse.path +
                             "}\n")
            if spec == VRML_EXT:
                if app.normalmap is not None:
                    stream.write("texture DEF normalmap_%s ImageTexture {\n" % app.normalmap.ident +
                                 "url \"%s\"\n" % app.normalmap.path +
                                 "}\n")
                if app.specular is not None:
                    stream.write("texture DEF specular_%s ImageTexture {\n" % app.specular.ident +
                                 "url \"%s\"\n" % app.specular.path +
                                 "}\n")
        stream.write("}\n")

    def writeShape(spec, stream, shape):
        stream.write("DEF OB_%s Transform {\n" % shape.ident +
                     "translation 0.0 0.0 0.0\n"
                     "rotation 1.0 0.0 0.0 0.0\n"
                     "scale 1.0 1.0 1.0\n"
                     "children [\n")
        stream.write("DEF ME_%s Group {\n" % shape.ident +
                     "children [\n"
                     "Shape {\n")
        writeMaterial(spec, stream, shape.material)
        stream.write("geometry IndexedFaceSet {\n"
                     "solid FALSE\n")
        if spec == VRML_EXT:
            if shape.smooth:
                stream.write("smooth TRUE\n")
            else:
                stream.write("smooth FALSE\n")
        stream.write("coord DEF FS_%s Coordinate {\n" % shape.ident +
                     "point [\n")
        for srcVert in shape.vertices:
            if shape.transform is not None:
                vert = shape.transform.value * numpy.matrix([[srcVert[0]], [srcVert[1]], [srcVert[2]], [1.]])
            else:
                vert = numpy.matrix([[srcVert[0]], [srcVert[1]], [srcVert[2]], [1.]])
            stream.write("%f %f %f\n" % tuple(vert[0:3]))
        stream.write("]\n"
                     "}\n"
                     "coordIndex [\n")
        for poly in shape.polygons:
            for index in poly:
                stream.write("%u " % index)
            stream.write("-1,\n")
        stream.write("]\n")
        #Write texture coordinates
        if (shape.material.diffuse, shape.material.normalmap, shape.material.specular) != (None, None, None):
            #stream.write("]\n");
            stream.write("texCoord TextureCoordinate {\n"
                         "point [\n");
            for vert in shape.texels:
                stream.write("%f %f,\n" % tuple(vert))
            stream.write("]\n"
                         "}\n");
            stream.write("texCoordIndex [\n");
            i = 0
            for poly in shape.polygons:
                for index in poly:
                    stream.write("%u " % i)
                    i += 1
                stream.write("-1\n")
            stream.write("]\n")
        stream.write("}\n")
        stream.write("}\n]\n}\n]\n}\n")


    out = open(path, "wb")
    out.write("#VRML V2.0 utf8\n#Created by vrml_export.py\n")
    for shape in data:
        writeShape(spec, out, shape)
    out.close()

#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# vrml_export.py
# Copyright (C) 2013 xent
# Project is distributed under the terms of the GNU General Public License v3.0

import math
import numpy
import time

import model

VRML_STRICT, VRML_KICAD = range(0, 2)

debugEnabled = False

def debug(text):
    if debugEnabled:
        print(text)

def store(data, path, spec=VRML_STRICT):
    exportedGroups, exportedMaterials = [], []

    def writeAppearance(spec, stream, material, level):
        def calcIntensity(ambient, diffuse):
            result = 0.
            for index in range(0, 3):
                if diffuse[index]:
                    result += ambient[index] / diffuse[index]
            return result / 3.

        ambIntensity = calcIntensity(material.color.ambient, material.color.diffuse)
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
        if transform is None:
            [vertices.extend(vertex) for vertex in geoVertices]
        else:
            [vertices.extend(transform.process(vertex)) for vertex in geoVertices]
        stream.write(" ".join(map(lambda x: str(round(x, 6)), vertices)))
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
        if any(texture is not None for texture in [material.diffuse, material.normal, material.specular]):
            texVertices, texPolygons = mesh.texture()

            stream.write("%stexCoord TextureCoordinate {\n" % ("\t" * (level + 1)))
            stream.write("%spoint [\n" % ("\t" * (level + 2)))
            stream.write("\t" * (level + 3))
            vertices = []
            [vertices.extend(vertex) for vertex in texVertices]
            stream.write(" ".join(map(lambda x: str(round(x, 6)), vertices)))
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
            translation = numpy.array([0., 0., 0.])
            rotation = numpy.array([1., 0., 0., 0.])
            scale = numpy.array([1., 1., 1.])
        else:
            translation = mesh.transform.value.getA()[:,3][0:3]
            translationMatrix = numpy.matrix([
                    [1., 0., 0., -translation[0]],
                    [0., 1., 0., -translation[1]],
                    [0., 0., 1., -translation[2]],
                    [0., 0., 0.,              1.]])
            translated = translationMatrix * mesh.transform.value

            scale = numpy.array([numpy.linalg.norm(translated.getA()[:,column][0:3]) for column in [0, 1, 2]])
            scaleMatrix = numpy.matrix([
                    [1. / scale[0],            0.,            0., 0.],
                    [           0., 1. / scale[1],            0., 0.],
                    [           0.,            0., 1. / scale[2], 0.],
                    [           0.,            0.,            0., 1.]])
            scaled = translated * scaleMatrix

            #Conversion from rotation matrix form to axis-angle form
            angle = math.acos(((scaled.trace() - 1.) - 1.) / 2.)

            if angle == 0.:
                rotation = numpy.array([1., 0., 0., 0.])
            else:
                skew = (scaled - scaled.transpose()).getA()
                vector = numpy.array([skew[2][1], skew[0][2], skew[1][0]])
                vector = (1. / (2. * math.sin(angle))) * vector
                vector = model.normalize(vector)

                if abs(angle) < math.pi:
                    rotation = numpy.array(vector.tolist() + [angle])
                else:
                    tensor = numpy.tensordot(vector, vector, 0)
                    values = numpy.array([tensor[2][1], tensor[0][2], tensor[1][0]])
                    vector = numpy.diag(tensor)
                    vector = model.normalize(vector)

                    posIndices, negIndices = [], []
                    for i in range(0, 3):
                        if values[i] < 0.:
                            negIndices.append(i)
                        elif values[i] > 0.:
                            posIndices.append(i)

                    if len(posIndices) == 1 and len(negIndices) == 2:
                        vector[posIndices[0]] *= -1.
                    elif len(posIndices) == 0 and len(negIndices) == 1:
                        vector[negIndices[0]] *= -1.

                    rotation = numpy.array(vector.tolist() + [angle])

            debug("Transform %s: translation %s, rotation %s, scale %s"
                    % (mesh.ident, str(translation), str(rotation), str(scale)))

        stream.write("%sDEF OB_%s Transform {\n" % ("\t" * level, mesh.ident))
        stream.write("%stranslation %f %f %f\n" % tuple(["\t" * (level + 1)] + translation.tolist()))
        stream.write("%srotation %f %f %f %f\n" % tuple(["\t" * (level + 1)] + rotation.tolist()))
        stream.write("%sscale %f %f %f\n" % tuple(["\t" * (level + 1)] + scale.tolist()))
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

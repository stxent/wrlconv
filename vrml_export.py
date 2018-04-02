#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# vrml_export.py
# Copyright (C) 2013 xent
# Project is distributed under the terms of the GNU General Public License v3.0

import math
import numpy
import time

try:
    import model
except ImportError:
    from . import model

VRML_STRICT, VRML_KICAD = range(0, 2)
debugEnabled = False

def debug(text):
    if debugEnabled:
        print(text)

def store(data, path, spec=VRML_STRICT):
    exportedGroups, exportedMaterials = [], []

    def encodeAppearance(spec, material, level):
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

            def encodeTexture(path, name, level):
                if name != "":
                    output += "%stexture DEF %s ImageTexture {\n" % ("\t" * level, name)
                else:
                    output += "%stexture %s {\n" % ("\t" * level)
                output += "%surl \"%s\"\n" % ("\t" * (level + 1), path)
                output += "%s}\n" % ("\t" * level)

            if material.diffuse is not None:
                output += encodeTexture(material.diffuse.path[0], material.diffuse.ident, level + 1)

        output += "%s}\n" % ("\t" * level)
        return output

    def encodeGeometry(spec, mesh, level):
        output = ""
        output += "%sgeometry IndexedFaceSet {\n" % ("\t" * level)

        appearance = mesh.appearance()
        output += "%ssolid %s\n" % ("\t" * (level + 1), "TRUE" if appearance.solid else "FALSE")
        output += "%ssmooth %s\n" % ("\t" * (level + 1), "TRUE" if appearance.smooth else "FALSE")

        geoVertices, geoPolygons = mesh.geometry()

        output += "%scoord DEF FS_%s Coordinate {\n" % ("\t" * (level + 1), mesh.ident)
        output += "%spoint [\n" % ("\t" * (level + 2))
        output += "\t" * (level + 3)
        vertices = []
        [vertices.extend(vertex) for vertex in geoVertices]
        output += " ".join([str(round(x, 6)) for x in vertices])
        output += "\n"
        output += "%s]\n" % ("\t" * (level + 2))
        output += "%s}\n" % ("\t" * (level + 1))

        output += "%scoordIndex [\n" % ("\t" * (level + 1))
        output += "\t" * (level + 2)
        indices = []
        [indices.extend(poly + [-1]) for poly in geoPolygons]
        output += " ".join([str(x) for x in indices])
        output += "\n"
        output += "%s]\n" % ("\t" * (level + 1))

        material = appearance.material
        if any(texture is not None for texture in [material.diffuse, material.normal, material.specular]):
            texVertices, texPolygons = mesh.texture()

            output += "%stexCoord TextureCoordinate {\n" % ("\t" * (level + 1))
            output += "%spoint [\n" % ("\t" * (level + 2))
            output += "\t" * (level + 3)
            vertices = []
            [vertices.extend(vertex) for vertex in texVertices]
            output += " ".join([str(round(x, 6)) for x in vertices])
            output += "\n"
            output += "%s]\n" % ("\t" * (level + 2))
            output += "%s}\n" % ("\t" * (level + 1))

            output += "%stexCoordIndex [\n" % ("\t" * (level + 1))
            output += "\t" * (level + 2)
            indices = []
            [indices.extend(poly + [-1]) for poly in texPolygons]
            output += " ".join([str(x) for x in indices])
            output += "\n"
            output += "%s]\n" % ("\t" * (level + 1))

        output += "%s}\n" % ("\t" * level)
        return output

    def encodeShape(spec, mesh, level):
        output = ""
        output += "%sShape {\n" % ("\t" * level)

        output += encodeAppearance(spec, mesh.appearance().material, level + 1)
        output += encodeGeometry(spec, mesh, level + 1)

        output += "%s}\n" % ("\t" * level)
        return output

    def encodeGroup(spec, mesh, level):
        output = ''
        alreadyExported = [group for group in exportedGroups if group.ident == mesh.ident]

        if len(alreadyExported) == 0:
            output += "%sDEF ME_%s Group {\n" % ("\t" * level, mesh.ident)
            output += "%schildren [\n" % ("\t" * (level + 1))
            output += encodeShape(spec, mesh, level + 2)
            output += "%s]\n" % ("\t" * (level + 1))
            output += "%s}\n" % ("\t" * level)
            exportedGroups.append(mesh)
        else:
            output += "%sUSE ME_%s\n" % ("\t" * level, mesh.ident)
            debug("Export: reused group %s" % mesh.ident)

        return output

    def encodeTransform(spec, mesh, level=0):
        output = ''
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

            # Conversion from rotation matrix form to axis-angle form
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

            debug('Transform %s: translation %s, rotation %s, scale %s'
                    % (mesh.ident, str(translation), str(rotation), str(scale)))

        output += "%sDEF OB_%s Transform {\n" % ("\t" * level, mesh.ident)
        output += "%stranslation %f %f %f\n" % tuple(["\t" * (level + 1)] + translation.tolist())
        output += "%srotation %f %f %f %f\n" % tuple(["\t" * (level + 1)] + rotation.tolist())
        output += "%sscale %f %f %f\n" % tuple(["\t" * (level + 1)] + scale.tolist())
        output += "%schildren [\n" % ("\t" * (level + 1))

        parent = mesh if mesh.parent is None else mesh.parent
        output += encodeGroup(spec, parent, level + 2)

        output += "%s]\n" % ("\t" * (level + 1))
        output += "%s}\n" % ("\t" * level)

        debug('Mesh exported in %f, name %s' % (time.time() - started, mesh.ident))
        return output

    out = open(path, 'wb')
    out.write('#VRML V2.0 utf8\n#Created by vrml_export.py\n'.encode('utf-8'))
    for shape in data:
        out.write(encodeTransform(spec, shape).encode('utf-8'))
    out.close()

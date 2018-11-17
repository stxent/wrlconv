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

debugEnabled = False

def debug(text):
    if debugEnabled:
        print(text)

def indent(level):
    return '\t' * level

def store(data, path):
    exportedGroups, exportedMaterials = [], []

    def encodeAppearance(material, level):
        def calcIntensity(ambient, diffuse):
            return sum([ambient[i] / diffuse[i] for i in range(0, 3) if diffuse[i] != 0.0]) / 3.0

        ambIntensity = min(calcIntensity(material.color.ambient, material.color.diffuse), 1.0)
        output = indent(level) + 'appearance Appearance {\n'

        if material in exportedMaterials:
            exported = exportedMaterials[exportedMaterials.index(material)]
            output += indent(level + 1) + 'material USE MA_{:s}\n'.format(exported.color.ident)
            debug('Export: reused material {:s} instead of {:s}'.format(exported.color.ident, material.color.ident))
        else:
            output += indent(level + 1) + 'material DEF MA_{:s} Material {{\n'.format(material.color.ident)
            output += indent(level + 2) + 'diffuseColor {:g} {:g} {:g}\n'.format(*material.color.diffuse)
            output += indent(level + 2) + 'ambientIntensity {:g}\n'.format(ambIntensity)
            output += indent(level + 2) + 'specularColor {:g} {:g} {:g}\n'.format(*material.color.specular)
            output += indent(level + 2) + 'emissiveColor {:g} {:g} {:g}\n'.format(*material.color.emissive)
            output += indent(level + 2) + 'shininess {:g}\n'.format(material.color.shininess)
            output += indent(level + 2) + 'transparency {:g}\n'.format(material.color.transparency)
            output += indent(level + 1) + '}\n'
            exportedMaterials.append(material)

            def encodeTexture(path, name, level):
                if name != '':
                    output += indent(level) + 'texture DEF {:s} ImageTexture {{\n'.format(name)
                else:
                    output += indent(level) + 'texture {:s} {{\n'.format(name)
                output += indent(level + 1) + 'url \'{:s}\'\n'.format(path)
                output += indent(level) + '}\n'

            if material.diffuse is not None:
                output += encodeTexture(material.diffuse.path[0], material.diffuse.ident, level + 1)

        output += indent(level) + '}\n'
        return output

    def encodeGeometry(mesh, level):
        output = indent(level) + 'geometry IndexedFaceSet {\n'

        appearance = mesh.appearance()
        output += indent(level + 1) + 'solid {:s}\n'.format('TRUE' if appearance.solid else 'FALSE')
        output += indent(level + 1) + 'smooth {:s}\n'.format('TRUE' if appearance.smooth else 'FALSE')

        geoVertices, geoPolygons = mesh.geometry()

        # Export vertices
        output += indent(level + 1) + 'coord DEF FS_{:s} Coordinate {{\n'.format(mesh.ident)
        output += indent(level + 2) + 'point [\n'
        output += indent(level + 3)
        vertices = []
        [vertices.extend(vertex) for vertex in geoVertices]
        output += ' '.join([str(round(x, 6)) for x in vertices])
        output += '\n'
        output += indent(level + 2) + ']\n'
        output += indent(level + 1) + '}\n'

        # Export polygons
        output += indent(level + 1) + 'coordIndex [\n'
        output += indent(level + 2)
        indices = []
        [indices.extend(poly + [-1]) for poly in geoPolygons]
        output += ' '.join([str(x) for x in indices])
        output += '\n'
        output += indent(level + 1) + ']\n'

        material = appearance.material
        if any(texture is not None for texture in [material.diffuse, material.normal, material.specular]):
            texVertices, texPolygons = mesh.texture()

            # Export texture vertices
            output += indent(level + 1) + 'texCoord TextureCoordinate {\n'
            output += indent(level + 2) + 'point [\n'
            output += indent(level + 3)
            vertices = []
            [vertices.extend(vertex) for vertex in texVertices]
            output += ' '.join([str(round(x, 6)) for x in vertices])
            output += '\n'
            output += indent(level + 2) + ']\n'
            output += indent(level + 1) + '}\n'

            # Export texture indices
            output += indent(level + 1) + 'texCoordIndex [\n'
            output += indent(level + 2)
            indices = []
            [indices.extend(poly + [-1]) for poly in texPolygons]
            output += ' '.join([str(x) for x in indices])
            output += '\n'
            output += indent(level + 1) + ']\n'

        output += indent(level) + '}\n'
        return output

    def encodeShape(mesh, level):
        output = indent(level) + 'Shape {\n'
        output += encodeAppearance(mesh.appearance().material, level + 1)
        output += encodeGeometry(mesh, level + 1)
        output += indent(level) + '}\n'
        return output

    def encodeGroup(mesh, level):
        output = ''
        alreadyExported = [group for group in exportedGroups if group.ident == mesh.ident]

        if len(alreadyExported) == 0:
            output += indent(level) + 'DEF ME_{:s} Group {{\n'.format(mesh.ident)
            output += indent(level + 1) + 'children [\n'
            output += encodeShape(mesh, level + 2)
            output += indent(level + 1) + ']\n'
            output += indent(level) + '}\n'
            exportedGroups.append(mesh)
        else:
            output += indent(level) + 'USE ME_{:s}\n'.format(mesh.ident)
            debug('Export: reused group {:s}'.format(mesh.ident))

        return output

    def encodeTransform(mesh, level=0):
        output = ''
        started = time.time()

        if mesh.transform is None:
            translation = numpy.array([0.0, 0.0, 0.0])
            rotation = numpy.array([1.0, 0.0, 0.0, 0.0])
            scale = numpy.array([1.0, 1.0, 1.0])
        else:
            translation = mesh.transform.matrix.getA()[:,3][0:3]
            translationMatrix = numpy.matrix([
                    [1.0, 0.0, 0.0, -translation[0]],
                    [0.0, 1.0, 0.0, -translation[1]],
                    [0.0, 0.0, 1.0, -translation[2]],
                    [0.0, 0.0, 0.0,             1.0]])
            translated = translationMatrix * mesh.transform.matrix

            scale = numpy.array([numpy.linalg.norm(translated.getA()[:,column][0:3]) for column in [0, 1, 2]])
            scaleMatrix = numpy.matrix([
                    [1.0 / scale[0],            0.0,            0.0, 0.0],
                    [           0.0, 1.0 / scale[1],            0.0, 0.0],
                    [           0.0,            0.0, 1.0 / scale[2], 0.0],
                    [           0.0,            0.0,            0.0, 1.0]])
            scaled = translated * scaleMatrix

            # Conversion from rotation matrix form to axis-angle form
            angle = math.acos(((scaled.trace() - 1.0) - 1.0) / 2.0)

            if angle == 0.:
                rotation = numpy.array([1.0, 0.0, 0.0, 0.0])
            else:
                skew = (scaled - scaled.transpose()).getA()
                vector = numpy.array([skew[2][1], skew[0][2], skew[1][0]])
                vector = (1.0 / (2.0 * math.sin(angle))) * vector
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
                        if values[i] < 0.0:
                            negIndices.append(i)
                        elif values[i] > 0.0:
                            posIndices.append(i)

                    if len(posIndices) == 1 and len(negIndices) == 2:
                        vector[posIndices[0]] *= -1.0
                    elif len(posIndices) == 0 and len(negIndices) == 1:
                        vector[negIndices[0]] *= -1.0

                    rotation = numpy.array(vector.tolist() + [angle])

            debug('Transform {:s}: translation {:s}, rotation {:s}, scale {:s}'.format(
                    mesh.ident, str(translation), str(rotation), str(scale)))

        output += indent(level) + 'DEF OB_{:s} Transform {{\n'.format(mesh.ident)
        output += indent(level + 1) + 'translation {:g} {:g} {:g}\n'.format(*translation)
        output += indent(level + 1) + 'rotation {:g} {:g} {:g} {:g}\n'.format(*rotation)
        output += indent(level + 1) + 'scale {:g} {:g} {:g}\n'.format(*scale)
        output += indent(level + 1) + 'children [\n'

        parent = mesh if mesh.parent is None else mesh.parent
        output += encodeGroup(parent, level + 2)

        output += indent(level + 1) + ']\n'
        output += indent(level) + '}\n'

        debug('Mesh exported in {:f}, name {:s}'.format(time.time() - started, mesh.ident))
        return output

    out = open(path, 'wb')
    out.write('#VRML V2.0 utf8\n#Created by vrml_export.py\n'.encode('utf-8'))
    for shape in data:
        out.write(encodeTransform(shape).encode('utf-8'))
    out.close()

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

def indent(level):
    return '\t' * level

def store(data, path):
    exportedMaterials = []

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

        output += indent(level) + '}\n'
        return output

    def encodeGeometry(mesh, transform, level):
        output = ''
        output += indent(level) + 'geometry IndexedFaceSet {\n'
        output += indent(level + 1) + 'solid FALSE\n'

        geoVertices, geoPolygons = mesh.geometry()

        # Export vertices
        output += indent(level + 1) + 'coord DEF FS_{:s} Coordinate {{\n'.format(mesh.ident)
        output += indent(level + 2) + 'point [\n'
        for vertex in geoVertices:
            output += '\t' + ' '.join([str(round(x, 6)) for x in transform.apply(vertex)]) + '\n'
        output += indent(level + 2) + ']\n'
        output += indent(level + 1) + '}\n'

        # Export polygons
        output += indent(level + 1) + 'coordIndex [\n'
        for poly in geoPolygons:
            output += '\t' + ' '.join([str(x) for x in poly]) + ' -1\n'
        output += indent(level + 1) + ']\n'

        output += indent(level) + '}\n'
        return output

    def encodeShape(mesh, transform, level):
        output = indent(level) + 'Shape {\n'
        output += encodeAppearance(mesh.appearance().material, level + 1)
        output += encodeGeometry(mesh, transform, level + 1)
        output += indent(level) + '}\n'
        return output

    def encodeGroup(mesh, topTransform, topName, level):
        output = ''
        transform = topTransform if mesh.transform is None else mesh.transform

        output += indent(level) + 'DEF ME_{:s}_{:s} Group {{\n'.format(topName, mesh.ident)
        output += indent(level + 1) + 'children [\n'
        output += encodeShape(mesh, transform, level + 2)
        output += indent(level + 1) + ']\n'
        output += indent(level) + '}\n'
        return output

    def encodeTransform(mesh, level=0):
        output = ''
        started = time.time()

        output += indent(level) + 'DEF OB_{:s} Transform {{\n'.format(mesh.ident)
        output += indent(level + 1) + 'translation 0.0 0.0 0.0\n'
        output += indent(level + 1) + 'rotation 1.0 0.0 0.0 0.0\n'
        output += indent(level + 1) + 'scale 1.0 1.0 1.0\n'
        output += indent(level + 1) + 'children [\n'

        parent = mesh if mesh.parent is None else mesh.parent
        if mesh.transform is not None:
            transform = mesh.transform
        else:
            transform = model.Transform()
        output += encodeGroup(parent, transform, mesh.ident, level + 2)

        output += indent(level + 1) + ']\n'
        output += indent(level) + '}\n'

        debug('Mesh exported in {:f}, name {:s}'.format(time.time() - started, mesh.ident))
        return output

    sortedData = [entry for entry in data if entry.appearance().material.color.transparency <= 0.001]
    sortedData += [entry for entry in data if entry.appearance().material.color.transparency > 0.001]

    out = open(path, 'wb')
    out.write('#VRML V2.0 utf8\n#Created by vrml_export_kicad.py\n'.encode('utf-8'))
    for shape in sortedData:
        out.write(encodeTransform(shape).encode('utf-8'))
    out.close()

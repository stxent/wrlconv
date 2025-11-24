#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# vrml_export.py
# Copyright (C) 2013 xent
# Project is distributed under the terms of the GNU General Public License v3.0

import time

try:
    import model
except ImportError:
    from . import model

DEBUG_ENABLED = False

def debug(text):
    if DEBUG_ENABLED:
        print(text)

def indent(level):
    return '\t' * level

def mangle_material_name(name):
    return name.replace('.', '')

def mangle_mesh_name(name):
    return name.replace('.', 'p')

def store(data, path):
    exported_materials = []

    def encode_appearance(material, level):
        def calc_intensity(ambient, diffuse):
            return sum([ambient[i] / diffuse[i] for i in range(0, 3) if diffuse[i] != 0.0]) / 3.0

        ambient_intensity = min(calc_intensity(material.color.ambient, material.color.diffuse), 1.0)
        output = indent(level) + 'appearance Appearance {\n'

        if material in exported_materials:
            exported = exported_materials[exported_materials.index(material)]
            output += indent(level + 1) + 'material USE MA_{:s}\n'.format(
                mangle_material_name(exported.color.ident))
            debug('Export: reused material {:s} instead of {:s}'.format(
                exported.color.ident, material.color.ident))
        else:
            output += indent(level + 1) + 'material DEF MA_{:s} Material {{\n'.format(
                mangle_material_name(material.color.ident))
            output += indent(level + 2) + 'diffuseColor {:g} {:g} {:g}\n'.format(
                *material.color.diffuse)
            output += indent(level + 2) + 'ambientIntensity {:g}\n'.format(ambient_intensity)
            output += indent(level + 2) + 'specularColor {:g} {:g} {:g}\n'.format(
                *material.color.specular)
            output += indent(level + 2) + 'emissiveColor {:g} {:g} {:g}\n'.format(
                *material.color.emissive)
            output += indent(level + 2) + 'shininess {:g}\n'.format(material.color.shininess)
            output += indent(level + 2) + 'transparency {:g}\n'.format(material.color.transparency)
            output += indent(level + 1) + '}\n'
            exported_materials.append(material)

        output += indent(level) + '}\n'
        return output

    def encode_geometry(mesh, transform, level):
        output = ''
        output += indent(level) + 'geometry IndexedFaceSet {\n'
        output += indent(level + 1) + 'solid FALSE\n'

        geo_vertices, geo_polygons = mesh.geometry()

        # Export vertices
        output += indent(level + 1) + 'coord DEF FS_{:s} Coordinate {{\n'.format(
            mangle_mesh_name(mesh.ident))
        output += indent(level + 2) + 'point [\n'
        for vertex in geo_vertices:
            output += '\t' + ' '.join([str(round(x, 6)) for x in transform.apply(vertex)]) + '\n'
        output += indent(level + 2) + ']\n'
        output += indent(level + 1) + '}\n'

        # Export polygons
        output += indent(level + 1) + 'coordIndex [\n'
        for poly in geo_polygons:
            output += '\t' + ' '.join([str(x) for x in poly]) + ' -1\n'
        output += indent(level + 1) + ']\n'

        output += indent(level) + '}\n'
        return output

    def encode_shape(mesh, transform, level):
        output = indent(level) + 'Shape {\n'
        output += encode_appearance(mesh.appearance().material, level + 1)
        output += encode_geometry(mesh, transform, level + 1)
        output += indent(level) + '}\n'
        return output

    def encode_group(mesh, top_transform, top_name, level):
        output = ''
        transform = top_transform if mesh.transform is None else mesh.transform

        output += indent(level) + 'DEF ME_{:s}_{:s} Group {{\n'.format(
            mangle_mesh_name(top_name), mangle_mesh_name(mesh.ident))
        output += indent(level + 1) + 'children [\n'
        output += encode_shape(mesh, transform, level + 2)
        output += indent(level + 1) + ']\n'
        output += indent(level) + '}\n'
        return output

    def encode_transform(mesh, level=0):
        output = ''
        started = time.time()

        output += indent(level) + 'DEF OB_{:s} Transform {{\n'.format(mangle_mesh_name(mesh.ident))
        output += indent(level + 1) + 'translation 0.0 0.0 0.0\n'
        output += indent(level + 1) + 'rotation 1.0 0.0 0.0 0.0\n'
        output += indent(level + 1) + 'scale 1.0 1.0 1.0\n'
        output += indent(level + 1) + 'children [\n'

        parent = mesh if mesh.parent is None else mesh.parent
        if mesh.transform is not None:
            transform = mesh.transform
        else:
            transform = model.Transform()
        output += encode_group(parent, transform, mesh.ident, level + 2)

        output += indent(level + 1) + ']\n'
        output += indent(level) + '}\n'

        debug('Mesh exported in {:f}, name {:s}'.format(time.time() - started, mesh.ident))
        return output

    entries = [entry for entry in data if entry.appearance().material.color.transparency <= 0.001]
    entries += [entry for entry in data if entry.appearance().material.color.transparency > 0.001]

    with open(path, 'wb') as out:
        out.write('#VRML V2.0 utf8\n#Created by vrml_export_kicad.py\n'.encode('utf-8'))
        for entry in entries:
            out.write(encode_transform(entry).encode('utf-8'))

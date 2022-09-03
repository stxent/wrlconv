#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# x3d_export.py
# Copyright (C) 2015 xent
# Project is distributed under the terms of the GNU General Public License v3.0

import itertools
import math
import os
import time
import numpy
from lxml import etree

try:
    import model
except ImportError:
    from . import model

DEBUG_ENABLED = False

def debug(text):
    if DEBUG_ENABLED:
        print(text)

def indent(element, level=0):
    i = '\n' + '\t' * level
    if element is not None:
        if not element.text or not element.text.strip():
            element.text = i + '\t'
        for entry in element:
            indent(entry, level + 1)
            if not entry.tail or not entry.tail.strip():
                entry.tail = i + '\t'
            if not entry.tail or not entry.tail.strip():
                entry.tail = i
    else:
        if level and (not element.tail or not element.tail.strip()):
            element.tail = i

def write_appearance(root, material, exported_materials):
    def calc_intensity(ambient, diffuse):
        result = 0.0
        for index in range(0, 3):
            if diffuse[index]:
                result += ambient[index] / diffuse[index]
        return result / 3.0

    appearance_node = etree.SubElement(root, 'Appearance')
    material_node = etree.SubElement(appearance_node, 'Material')

    ambient_intensity = calc_intensity(material.color.ambient, material.color.diffuse)
    if material in exported_materials:
        exported = exported_materials[exported_materials.index(material)]
        material_node.attrib['USE'] = 'MA_{:s}'.format(exported.color.ident)
        debug('Export: reused material {:s} instead of {:s}'.format(
            exported.color.ident, material.color.ident))
    else:
        material_node.attrib['DEF'] = 'MA_{:s}'.format(material.color.ident)
        material_node.attrib['diffuseColor'] = '{:g} {:g} {:g}'.format(
            *material.color.diffuse)
        material_node.attrib['specularColor'] = '{:g} {:g} {:g}'.format(
            *material.color.specular)
        material_node.attrib['emissiveColor'] = '{:g} {:g} {:g}'.format(
            *material.color.emissive)
        material_node.attrib['ambientIntensity'] = str(ambient_intensity)
        material_node.attrib['shininess'] = str(material.color.shininess)
        material_node.attrib['transparency'] = str(material.color.transparency)
        exported_materials.append(material)

    write_texture(appearance_node, material)

def write_geometry(root, mesh):
    appearance = mesh.appearance()
    geo_vertices, geo_polygons = mesh.geometry()

    faceset = etree.SubElement(root, 'IndexedFaceSet')
    faceset.attrib['solid'] = 'true' if appearance.solid else 'false'
    indices = list(itertools.chain.from_iterable([poly + [-1] for poly in geo_polygons]))
    faceset.attrib['coordIndex'] = ' '.join([str(x) for x in indices])

    geo_coords = etree.SubElement(faceset, 'Coordinate')
    geo_coords.attrib['DEF'] = 'FS_{:s}'.format(mesh.ident)
    vertices = list(itertools.chain.from_iterable(geo_vertices))
    geo_coords.attrib['point'] = ' '.join([str(round(x, 6)) for x in vertices])

    material = appearance.material
    if any(texture is not None for texture in
           [material.diffuse, material.normal, material.specular]):
        tex_vertices, tex_polygons = mesh.texture()
        tex_coords = etree.SubElement(faceset, 'TextureCoordinate')

        vertices = list(itertools.chain.from_iterable(tex_vertices))
        tex_coords.attrib['point'] = ' '.join([str(round(x, 6)) for x in vertices])

        indices = list(itertools.chain.from_iterable([poly + [-1] for poly in tex_polygons]))
        faceset.attrib['texCoordIndex'] = ' '.join([str(x) for x in indices])

def write_shape(root, mesh, exported_materials):
    shape = etree.SubElement(root, 'Shape')
    write_appearance(shape, mesh.appearance().material, exported_materials)
    write_geometry(shape, mesh)

def write_group(root, mesh, exported_groups, exported_materials):
    already_exported = [group for group in exported_groups if group.ident == mesh.ident]

    group = etree.SubElement(root, 'Group')
    if not already_exported:
        group.attrib['DEF'] = 'ME_{:s}'.format(mesh.ident)
        write_shape(group, mesh, exported_materials)
        exported_groups.append(mesh)
    else:
        group.attrib['USE'] = 'ME_{:s}'.format(mesh.ident)
        debug('Export: reused group {:s}'.format(mesh.ident))

def write_texture(root, material):
    if material.diffuse is not None and material.normal is None and material.specular is None:
        texture_node = etree.SubElement(root, 'ImageTexture')
        texture_node.attrib['DEF'] = material.diffuse.ident
        texture_node.attrib['url'] = '\'{:s}\' \'{:s}\''.format(*material.diffuse.path)
    else:
        chain, modes, sources = [], [], []
        if material.normal is not None:
            chain.append(material.normal)
            modes.append('DOTPRODUCT3')
            sources.append('DIFFUSE')
        if material.diffuse is not None:
            chain.append(material.diffuse)
            modes.append('MODULATE')
            if material.normal is not None:
                sources.append('')
            else:
                sources.append('DIFFUSE')
        if material.specular is not None:
            chain.append(material.specular)
            modes.append('MODULATE')
            sources.append('SPECULAR')

        if chain:
            multi_texture_node = etree.SubElement(root, 'MultiTexture')
            multi_texture_node.attrib['mode'] = ' '.join(
                ['\'{:s}\''.format(x) for x in modes])
            multi_texture_node.attrib['source'] = ' '.join(
                ['\'{:s}\''.format(x) for x in sources])

            for entry in chain:
                texture_node = etree.SubElement(multi_texture_node, 'ImageTexture')
                texture_node.attrib['DEF'] = entry.ident
                texture_node.attrib['url'] = '\'{:s}\' \'{:s}\''.format(*entry.path)

def write_transform(root, mesh, exported_groups, exported_materials):
    started = time.time()

    if mesh.transform is None:
        translation = numpy.array([0.0, 0.0, 0.0])
        rotation = numpy.array([1.0, 0.0, 0.0, 0.0])
        scale = numpy.array([1.0, 1.0, 1.0])
    else:
        translation = mesh.transform.matrix[:,3][0:3]
        translation_matrix = numpy.array([
            [1.0, 0.0, 0.0, -translation[0]],
            [0.0, 1.0, 0.0, -translation[1]],
            [0.0, 0.0, 1.0, -translation[2]],
            [0.0, 0.0, 0.0,             1.0]])
        translated = numpy.matmul(translation_matrix, mesh.transform.matrix)

        scale = numpy.array([numpy.linalg.norm(
            translated[:,column][0:3]) for column in [0, 1, 2]])
        scale_matrix = numpy.array([
            [1.0 / scale[0],            0.0,            0.0, 0.0],
            [           0.0, 1.0 / scale[1],            0.0, 0.0],
            [           0.0,            0.0, 1.0 / scale[2], 0.0],
            [           0.0,            0.0,            0.0, 1.0]])
        scaled = numpy.matmul(translated, scale_matrix)

        # Conversion from rotation matrix form to axis-angle form
        angle = math.acos(((scaled.trace() - 1.0) - 1.0) / 2.0)

        if angle == 0.0:
            rotation = numpy.array([1.0, 0.0, 0.0, 0.0])
        else:
            skew = scaled - scaled.transpose()
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

                pos_indices, neg_indices = [], []
                for i in range(0, 3):
                    if values[i] < 0.0:
                        neg_indices.append(i)
                    elif values[i] > 0.0:
                        pos_indices.append(i)

                if len(pos_indices) == 1 and len(neg_indices) == 2:
                    vector[pos_indices[0]] *= -1.0
                elif not pos_indices and len(neg_indices) == 1:
                    vector[neg_indices[0]] *= -1.0

                rotation = numpy.array(vector.tolist() + [angle])

        debug('Transform {:s}: translation {:s}, rotation {:s}, scale {:s}'.format(
            mesh.ident, str(translation), str(rotation), str(scale)))

    transform = etree.SubElement(root, 'Transform')
    transform.attrib['DEF'] = 'OB_{:s}'.format(mesh.ident)
    transform.attrib['translation'] = '{:g} {:g} {:g}'.format(*translation)
    transform.attrib['scale'] = '{:g} {:g} {:g}'.format(*scale)
    transform.attrib['rotation'] = '{:g} {:g} {:g} {:g}'.format(*rotation)

    parent = mesh if mesh.parent is None else mesh.parent
    write_group(transform, parent, exported_groups, exported_materials)

    debug('Mesh exported in {:f}, name {:s}'.format(time.time() - started, mesh.ident))

def convert(data, name=None):
    exported_groups, exported_materials = [], []
    filename = 'unnamed.x3d' if name is None else name

    doctype = '<!DOCTYPE X3D PUBLIC "ISO//Web3D//DTD X3D 3.0//EN" "http://www.web3d.org/specifications/x3d-3.0.dtd">'

    root = etree.Element('X3D')
    root.attrib['version'] = '3.0'
    root.attrib['profile'] = 'Immersive'
    # root.attrib['xmlns:xsd'] = 'http://www.w3.org/2001/XMLSchema-instance'
    # root.attrib['xsd:noNamespaceSchemaLocation'] = 'http://www.web3d.org/specifications/x3d-3.0.xsd'

    head = etree.SubElement(root, 'head')
    etree.SubElement(head, 'meta', name='filename', content=filename)
    etree.SubElement(head, 'meta', name='generator', content='x3d_export.py')

    scene = etree.SubElement(root, 'Scene')

    for shape in data:
        write_transform(scene, shape, exported_groups, exported_materials)

    indent(root)
    payload = etree.tostring(root, pretty_print=True, xml_declaration=True, encoding='UTF-8',
                             doctype=doctype)
    # Replace quotes to match X3D specification
    payload = payload.decode('utf-8')
    payload = payload.replace('"&quot;', '\'"').replace('&quot;"', '"\'').replace('&quot;', '"')

    return payload

def store(data, path):
    payload = convert(data, os.path.basename(path))

    with open(path, 'wb') as out:
        out.write(payload.encode('utf-8'))

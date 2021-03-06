#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# x3d_export.py
# Copyright (C) 2015 xent
# Project is distributed under the terms of the GNU General Public License v3.0

import math
import os
import numpy
import time
from lxml import etree

try:
    import model
except ImportError:
    from . import model

debugEnabled = False

def debug(text):
    if debugEnabled:
        print(text)

def indent(element, level=0):
    i = '\n' + '\t' * level
    if len(element):
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

def store(data, path):
    exportedGroups, exportedMaterials = [], []

    def writeTexture(root, material):
        if material.diffuse is not None and material.normal is None and material.specular is None:
            textureNode = etree.SubElement(root, 'ImageTexture')
            textureNode.attrib['DEF'] = material.diffuse.ident
            textureNode.attrib['url'] = '\'{:s}\' \'{:s}\''.format(*material.diffuse.path)
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

            if len(chain) > 0:
                multiTextureNode = etree.SubElement(root, 'MultiTexture')
                multiTextureNode.attrib['mode'] = ' '.join(['\'{:s}\''.format(x) for x in modes])
                multiTextureNode.attrib['source'] = ' '.join(['\'{:s}\''.format(x) for x in sources])

                for entry in chain:
                    textureNode = etree.SubElement(multiTextureNode, 'ImageTexture')
                    textureNode.attrib['DEF'] = entry.ident
                    textureNode.attrib['url'] = '\'{:s}\' \'{:s}\''.format(*entry.path)

    def writeAppearance(root, material):
        def calcIntensity(ambient, diffuse):
            result = 0.
            for index in range(0, 3):
                if diffuse[index]:
                    result += ambient[index] / diffuse[index]
            return result / 3.0

        appearanceNode = etree.SubElement(root, 'Appearance')
        materialNode = etree.SubElement(appearanceNode, 'Material')

        ambIntensity = calcIntensity(material.color.ambient, material.color.diffuse)
        if material in exportedMaterials:
            exported = exportedMaterials[exportedMaterials.index(material)]
            materialNode.attrib['USE'] = 'MA_{:s}'.format(exported.color.ident)
            debug('Export: reused material {:s} instead of {:s}'.format(exported.color.ident, material.color.ident))
        else:
            materialNode.attrib['DEF'] = 'MA_{:s}'.format(material.color.ident)
            materialNode.attrib['diffuseColor'] = '{:g} {:g} {:g}'.format(*material.color.diffuse)
            materialNode.attrib['specularColor'] = '{:g} {:g} {:g}'.format(*material.color.specular)
            materialNode.attrib['emissiveColor'] = '{:g} {:g} {:g}'.format(*material.color.emissive)
            materialNode.attrib['ambientIntensity'] = str(ambIntensity)
            materialNode.attrib['shininess'] = str(material.color.shininess)
            materialNode.attrib['transparency'] = str(material.color.transparency)
            exportedMaterials.append(material)

        writeTexture(appearanceNode, material)

    def writeGeometry(root, mesh):
        appearance = mesh.appearance()
        geoVertices, geoPolygons = mesh.geometry()

        faceset = etree.SubElement(root, 'IndexedFaceSet')
        faceset.attrib['solid'] = 'true' if appearance.solid else 'false'
        indices = []
        [indices.extend(poly + [-1]) for poly in geoPolygons]
        faceset.attrib['coordIndex'] = ' '.join([str(x) for x in indices])

        geoCoords = etree.SubElement(faceset, 'Coordinate')
        geoCoords.attrib['DEF'] = 'FS_{:s}'.format(mesh.ident)
        vertices = []
        [vertices.extend(vertex) for vertex in geoVertices]
        geoCoords.attrib['point'] = ' '.join([str(round(x, 6)) for x in vertices])

        material = appearance.material
        if any(texture is not None for texture in [material.diffuse, material.normal, material.specular]):
            texVertices, texPolygons = mesh.texture()
            texCoords = etree.SubElement(faceset, 'TextureCoordinate')

            vertices = []
            [vertices.extend(vertex) for vertex in texVertices]
            texCoords.attrib['point'] = ' '.join([str(round(x, 6)) for x in vertices])

            indices = []
            [indices.extend(poly + [-1]) for poly in texPolygons]
            faceset.attrib['texCoordIndex'] = ' '.join([str(x) for x in indices])


    def writeShape(root, mesh):
        shape = etree.SubElement(root, 'Shape')
        writeAppearance(shape, mesh.appearance().material)
        writeGeometry(shape, mesh)

    def writeGroup(root, mesh):
        alreadyExported = [group for group in exportedGroups if group.ident == mesh.ident]

        group = etree.SubElement(root, 'Group')
        if len(alreadyExported) == 0:
            group.attrib['DEF'] = 'ME_{:s}'.format(mesh.ident)
            writeShape(group, mesh)
            exportedGroups.append(mesh)
        else:
            group.attrib['USE'] = 'ME_{:s}'.format(mesh.ident)
            debug('Export: reused group {:s}'.format(mesh.ident))

    def writeTransform(root, mesh):
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

            if angle == 0.0:
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

        transform = etree.SubElement(root, 'Transform')
        transform.attrib['DEF'] = 'OB_{:s}'.format(mesh.ident)
        transform.attrib['translation'] = '{:g} {:g} {:g}'.format(*translation)
        transform.attrib['scale'] = '{:g} {:g} {:g}'.format(*scale)
        transform.attrib['rotation'] = '{:g} {:g} {:g} {:g}'.format(*rotation)

        parent = mesh if mesh.parent is None else mesh.parent
        writeGroup(transform, parent)

        debug('Mesh exported in {:f}, name {:s}'.format(time.time() - started, mesh.ident))

    doctype = '<!DOCTYPE X3D PUBLIC "ISO//Web3D//DTD X3D 3.0//EN" "http://www.web3d.org/specifications/x3d-3.0.dtd">'
    filename = os.path.basename(path)

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
        writeTransform(scene, shape)

    indent(root)
    payload = etree.tostring(root, pretty_print=True, xml_declaration=True, encoding='UTF-8', doctype=doctype)
    # Replace quotes to match X3D specification
    payload = payload.decode('utf-8').replace('"&quot;', '\'"').replace('&quot;"', '"\'').replace('&quot;', '"')

    out = open(path, 'wb')
    out.write(payload.encode('utf-8'))
    out.close()

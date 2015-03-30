#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# x3d_export.py
# Copyright (C) 2015 xent
# Project is distributed under the terms of the GNU General Public License v3.0

import os
import numpy
import time
from lxml import etree

import model

debugEnabled = False

def debug(text):
    if debugEnabled:
        print(text)

def store(data, path):
    exportedGroups, exportedMaterials = [], []

    def writeAppearance(root, material):
        appearanceNode = etree.SubElement(root, "Appearance")
        materialNode = etree.SubElement(appearanceNode, "Material")
        
        ambIntensity = sum(map(lambda i: material.color.ambient[i] / material.color.diffuse[i] / 3., range(0, 3)))

        if material in exportedMaterials:
            materialNode.attrib["USE"] = "MA_%s" % material.color.ident
            debug("Export: reused material %s" % material.color.ident)
        else:
            materialNode.attrib["DEF"] = "MA_%s" % material.color.ident
            materialNode.attrib["diffuseColor"] = "%f %f %f" % tuple(material.color.diffuse)
            materialNode.attrib["specularColor"] = "%f %f %f" % tuple(material.color.specular)
            materialNode.attrib["emissiveColor"] = "%f %f %f" % tuple(material.color.emissive)
            materialNode.attrib["ambientIntensity"] = str(ambIntensity)
            materialNode.attrib["shininess"] = str(material.color.shininess)
            materialNode.attrib["transparency"] = str(material.color.transparency)
            exportedMaterials.append(material)

            if material.diffuse is not None:
                textureNode = etree.SubElement(appearanceNode, "ImageTexture")
                textureNode.attrib["DEF"] = material.diffuse.ident
                #FIXME
                textureNode.attrib["url"] = "\"%s\" \"%s\"" % tuple(material.diffuse.path)

                textureTransformNode = etree.SubElement(appearanceNode, "ImageTexture")
                textureTransformNode.attrib["translation"] = "0.0 0.0"
                textureTransformNode.attrib["scale"] = "1.0 1.0"
                textureTransformNode.attrib["rotation"] = "0.0"

    def writeGeometry(root, mesh):
        appearance = mesh.appearance()
        geoVertices, geoPolygons = mesh.geometry()

        faceset = etree.SubElement(root, "IndexedFaceSet")

        faceset.attrib["solid"] = "true" if appearance.solid else "false"
        indices = []
        [indices.extend(poly) for poly in map(lambda poly: poly + [-1], geoPolygons)]
        faceset.attrib["coordIndex"] = " ".join(map(str, indices))

        geoCoords = etree.SubElement(faceset, "Coordinate")
        geoCoords.attrib["DEF"] = "FS_%s" % mesh.ident
        vertices = []
        for srcVert in geoVertices:
            vert = numpy.matrix([[srcVert[0]], [srcVert[1]], [srcVert[2]], [1.]])
            if mesh.transform is not None:
                vert = mesh.transform.value * vert
            vertices.extend([vert[0, 0], vert[1, 0], vert[2, 0]])
        geoCoords.attrib["point"] = " ".join(map(str, vertices))

        material = appearance.material
        if any(texture is not None for texture in [material.diffuse, material.normalmap, material.specular]):
            texVertices, texPolygons = mesh.texture()
            texCoords = etree.SubElement(faceset, "TextureCoordinate")

            vertices = []
            [vertices.extend(vertex) for vertex in texVertices]
            texCoords.attrib["point"] = " ".join(map(str, vertices))

            indices = []
            [indices.extend(poly) for poly in map(lambda poly: poly + [-1], texPolygons)]
            faceset.attrib["texCoordIndex"] = " ".join(map(str, indices))


    def writeShape(root, mesh):
        shape = etree.SubElement(root, "Shape")
        writeAppearance(shape, mesh.appearance().material)
        writeGeometry(shape, mesh)

    def writeGroup(root, mesh):
        alreadyExported = filter(lambda group: group.ident == mesh.ident, exportedGroups)

        group = etree.SubElement(root, "Group")
        if len(alreadyExported) == 0:
            group.attrib["DEF"] = "ME_%s" % mesh.ident
            writeShape(group, mesh)
            exportedGroups.append(mesh)
        else:
            group.attrib["USE"] = "ME_%s" % mesh.ident
            debug("Export: reused group %s" % mesh.ident)

    def writeTransform(root, mesh):
        started = time.time()

        if mesh.transform is None:
            translation = [0., 0., 0.]
        else:
            column = mesh.transform.value[:,3][0:3]
            translation = [column[0], column[1], column[2]]
        
        transform = etree.SubElement(root, "Transform")
        transform.attrib["DEF"] = "OB_%s" % mesh.ident
        transform.attrib["translation"] = "%f %f %f" % tuple(translation)
        transform.attrib["scale"] = "1.0 1.0 1.0"
        transform.attrib["rotation"] = "1.0 0.0 0.0 0.0"

        parent = mesh if mesh.parent is None else mesh.parent
        writeGroup(transform, parent)

        debug("Mesh exported in %f, name %s" % (time.time() - started, mesh.ident))

    doctype = '<!DOCTYPE X3D PUBLIC "ISO//Web3D//DTD X3D 3.0//EN" "http://www.web3d.org/specifications/x3d-3.0.dtd">'
    fileName = os.path.basename(path)

    root = etree.Element("X3D")
    root.attrib["version"] = "3.0"
    root.attrib["profile"] = "Immersive"
#    root.attrib["xmlns:xsd"] = "http://www.w3.org/2001/XMLSchema-instance"
#    root.attrib["xsd:noNamespaceSchemaLocation"] = "http://www.web3d.org/specifications/x3d-3.0.xsd"

    head = etree.SubElement(root, "head")
    etree.SubElement(head, "meta", name="filename", content=fileName)
    etree.SubElement(head, "meta", name="generator", content="x3d_export.py")

    scene = etree.SubElement(root, "Scene")
    for shape in data:
        writeTransform(scene, shape)

    out = open(path, "wb")
    out.write(etree.tostring(root, pretty_print=True, xml_declaration=True, encoding="UTF-8", doctype=doctype))
    out.close()

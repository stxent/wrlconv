#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# x3d_import.py
# Copyright (C) 2015 xent
# Project is distributed under the terms of the GNU General Public License v3.0

import numpy
import re
from xml.parsers import expat

import model

debugEnabled = False

def debug(text):
    if debugEnabled:
        print(text)


class X3dEntry:
    def __init__(self, parent=None):
        self.ident = None
        self.parent = parent
        self.name = ""
        self.ancestors = []
        self.level = self.parent.level + 2 if self.parent is not None else 0

    def demangled(self):
        return self.name

    def parse(self, attributes):
        pass

    def parseAttributes(self, attributes):
        if "DEF" in attributes.keys():
            self.name = attributes["DEF"]
            debug("%sEntry name %s" % (' ' * self.level, self.name))
        self.parse(attributes)


class X3dScene(X3dEntry):
    def __init__(self):
        X3dEntry.__init__(self)
        self.transform = model.Transform()

    def extract(self):
        exportedMaterials, exportedMeshes = [], []

        def createMesh(geometry, appearance, name):
            #Create abstract mesh object
            mesh = model.Mesh(name=name)
            mesh.geoPolygons = geometry.geoPolygons
            mesh.texPolygons = geometry.texPolygons
            if appearance is not None:
                newMaterial = appearance.squash()
                materials = filter(lambda mat: mat == newMaterial, exportedMaterials)
                if len(materials) > 0:
                    debug("Squash: reused material %s" % materials[0].color.ident)
                    mesh.visualAppearance.material = materials[0]
                else:
                    mesh.visualAppearance.material = newMaterial
                    exportedMaterials.append(newMaterial)
            mesh.visualAppearance.smooth = geometry.smooth
            mesh.visualAppearance.solid = geometry.solid
            for subentry in geometry.ancestors:
                if isinstance(subentry, X3dGeoCoords):
                    mesh.geoVertices = subentry.vertices
                elif isinstance(subentry, X3dTexCoords):
                    mesh.texVertices = subentry.vertices
            return mesh

        def reindexMesh(mesh):
            used = []
            [used.append(i) for poly in mesh.geoPolygons for i in poly if i not in used]
            used.sort()

            vertices = [mesh.geoVertices[i] for i in used]
            translated = dict(zip(used, range(0, len(vertices))))
            polygons = map(lambda poly: [translated[i] for i in poly], mesh.geoPolygons)

            debug("Reindex: mesh %s, %d polygons, from %d to %d vertices" % (mesh.ident, len(polygons),
                    len(mesh.geoVertices), len(vertices)))
            mesh.geoPolygons = polygons
            mesh.geoVertices = vertices

        def squash(entry, transform, name=[]):
            if isinstance(entry, X3dTransform):
                parts = []
                for i in range(0, len(entry.ancestors)):
                    shape = entry.ancestors[i]
                    demangled = entry.demangled() if entry.demangled() != "" else entry.name
                    subname = name + [demangled] if demangled != "" else name
                    subname[-1] += "_" + str(i) if len(entry.ancestors) > 1 else ""
                    parts.extend(squash(shape, transform * entry.transform, subname))
                return parts
            elif isinstance(entry, X3dShape):
                appearance, geometry = None, None
                for subentry in entry.ancestors:
                    if isinstance(subentry, X3dAppearance):
                        appearance = subentry
                    if isinstance(subentry, X3dGeometry):
                        geometry = subentry
                if geometry is not None:
                    alreadyExported = None
                    for mesh in exportedMeshes:
                        if mesh.ident == name[-1]:
                            alreadyExported = mesh
                            break
                    if alreadyExported is not None:
                        debug("Squash: reused mesh %s" % name[-1])
                        #Create concrete shape
                        currentMesh = model.Mesh(parent=alreadyExported, name=name[0])
                        currentMesh.transform = transform
                        return [currentMesh]
                    else:
                        mesh = createMesh(geometry, appearance, name[-1])
                        if entry.parent is not None and entry.parent.name != "" and len(entry.parent.ancestors) > 1:
                            reindexMesh(mesh)
                        exportedMeshes.append(mesh)
                        #Create concrete shape
                        currentMesh = model.Mesh(parent=mesh, name=name[0])
                        currentMesh.transform = transform
                        return [currentMesh]
            return []
        entries = []
        map(entries.extend, map(lambda x: squash(x, self.transform), self.ancestors))
        return entries


class X3dTransform(X3dEntry):
    def __init__(self, parent):
        X3dEntry.__init__(self, parent)
        self.transform = model.Transform()
        
    def parse(self, attributes):
        def getValues(string):
            return map(float, string.split(" "))
        if "rotation" in attributes.keys():
            values = getValues(attributes["rotation"])
            vector, angle = model.normalize(values[0:3]), values[3]
            vector = model.normalize(vector)
            self.transform.rotate(vector, angle)
        if "scale" in attributes.keys():
            self.transform.scale(getValues(attributes["scale"]))
        if "translation" in attributes.keys():
            self.transform.translate(getValues(attributes["translation"]))

    def demangled(self):
        #Demangle Blender names
        name = self.name.replace("OB_", "").replace("group_ME_", "")
        name = name.replace("_ifs", "").replace("_TRANSFORM", "")
        #Demangle own names
        name = name.replace("ME_", "")

        return name


class X3dShape(X3dEntry):
    def __init__(self, parent):
        X3dEntry.__init__(self, parent)


class X3dGeometry(X3dEntry):
    POLYGON_PATTERN = re.compile("([ ,\t\d]+)-1", re.I | re.S)

    def __init__(self, parent):
        X3dEntry.__init__(self, parent)
        self.geoPolygons = []
        self.texPolygons = []
        self.smooth = False
        self.solid = True

    def parse(self, attributes):
        if "solid" in attributes.keys():
            self.solid = True if attributes["solid"] == "true" else False
        if "smooth" in attributes.keys():
            self.smooth = True if attributes["smooth"] == "true" else False

        def parsePolygons(string):
            chunks = filter(lambda poly: len(poly) > 0, map(lambda entry: entry.strip(), string.split("-1")))
            polygons = map(lambda poly: map(int, filter(lambda vertex: len(vertex) > 0, poly.split(" "))), chunks)
            return polygons

        if "coordIndex" in attributes.keys():
            self.geoPolygons = parsePolygons(attributes["coordIndex"])
            debug("%sFound %u polygons" % (' ' * self.level, len(self.geoPolygons)))
        if "texCoordIndex" in attributes.keys():
            self.texPolygons = parsePolygons(attributes["texCoordIndex"])
            debug("%sFound %u texture polygons" % (' ' * self.level, len(self.texPolygons)))


class X3dCoords(X3dEntry):
    def __init__(self, parent, size):
        X3dEntry.__init__(self, parent)
        self.size = size
        self.vertices = None

    def parse(self, attributes):
        if "point" in attributes.keys():
            points = map(float, filter(lambda vertex: len(vertex) > 0, attributes["point"].split(" ")))
            vertices = []
            [vertices.append(numpy.array(points[i * self.size:(i + 1) * self.size]))
                    for i in range(0, len(points) / self.size)]
            self.vertices = vertices
            debug("%sFound %u vertices, width %u" % (' ' * self.level, len(self.vertices), self.size))


class X3dGeoCoords(X3dCoords):
    def __init__(self, parent):
        X3dCoords.__init__(self, parent, 3)


class X3dTexCoords(X3dCoords):
    def __init__(self, parent):
        X3dCoords.__init__(self, parent, 2)


class X3dAppearance(X3dEntry):
    def __init__(self, parent):
        X3dEntry.__init__(self, parent)
        #Limited support for image textures: TextureTransform nodes unsupported

    def squash(self):
        material = model.Material()
        for entry in self.ancestors:
            if isinstance(entry, X3dMaterial):
                material.color = entry.color
            elif isinstance(entry, X3dTexture):
                material.diffuse = entry.texture
            elif isinstance(entry, X3dMultiTexture):
                if max(entry.mapping.values()) >= len(entry.ancestors):
                    raise Exception() #Texture index is out of range
                if entry.mapping["diffuse"] is not None:
                    material.diffuse = entry.ancestors[entry.mapping["diffuse"]].texture
                if entry.mapping["normal"] is not None:
                    material.normal = entry.ancestors[entry.mapping["normal"]].texture
                if entry.mapping["specular"] is not None:
                    material.specular = entry.ancestors[entry.mapping["specular"]].texture
        return material


class X3dMaterial(X3dEntry):
    IDENT = 0

    def __init__(self, parent):
        X3dEntry.__init__(self, parent)
        self.color = model.Material.Color("DefaultX3dMaterial_%u" % X3dMaterial.IDENT)
        X3dMaterial.IDENT += 1

    def parse(self, attributes):
        def getValues(string):
            return numpy.array(list(map(float, string.split(" "))))

        self.color.ident = self.demangled()
        if "shininess" in attributes.keys():
            self.color.shininess = float(attributes["shininess"])
        if "transparency" in attributes.keys():
            self.color.transparency = float(attributes["transparency"])
        if "diffuseColor" in attributes.keys():
            self.color.diffuse = getValues(attributes["diffuseColor"])
        if "emissiveColor" in attributes.keys():
            self.color.emissive = getValues(attributes["emissiveColor"])
        if "specularColor" in attributes.keys():
            self.color.specular = getValues(attributes["specularColor"])
        if "ambientIntensity" in attributes.keys():
            self.color.ambient = self.color.diffuse * float(attributes["ambientIntensity"])

        debug("%sMaterial properties:" % (' ' * self.level))
        debug("%sShininess:      %f" % (' ' * (self.level + 1), self.color.shininess))
        debug("%sTransparency:   %f" % (' ' * (self.level + 1), self.color.transparency))
        debug("%sDiffuse Color:  %f, %f, %f" % (' ' * (self.level + 1),
                self.color.diffuse[0], self.color.diffuse[1], self.color.diffuse[2]))
        debug("%sEmissive Color: %f, %f, %f" % (' ' * (self.level + 1),
                self.color.emissive[0], self.color.emissive[1], self.color.emissive[2]))
        debug("%sSpecular Color: %f, %f, %f" % (' ' * (self.level + 1),
                self.color.specular[0], self.color.specular[1], self.color.specular[2]))
        debug("%sAmbient Color:  %f, %f, %f" % (' ' * (self.level + 1),
                self.color.ambient[0], self.color.ambient[1], self.color.ambient[2]))

    def demangled(self):
        #Demangle Blender names
        return self.name.replace("MA_", "")


class X3dTexture(X3dEntry):
    IDENT = 0

    def __init__(self, parent):
        X3dEntry.__init__(self, parent)
        self.texture = model.Material.Texture("", "DefaultX3dTexture_%u" % X3dTexture.IDENT)
        X3dTexture.IDENT += 1

    def parse(self, attributes):
        if "url" in attributes.keys():
            self.texture.path = map(lambda x: x.replace("\"", ""), attributes["url"].split(" "))


class X3dMultiTexture(X3dEntry):
    def __init__(self, parent):
        X3dEntry.__init__(self, parent)
        #Limited support for this type of node
        self.mapping = {"diffuse": None, "normal": None, "specular": None}

    def parse(self, attributes):
        modes, sources = [], []
        if "source" in attributes.keys():
            sources = map(lambda x: x.replace("\"", ""), attributes["source"].split(" "))
        if "mode" in attributes.keys():
            modes = map(lambda x: x.replace("\"", ""), attributes["mode"].split(" "))
        texCount = max(len(modes), len(sources))
        modes.extend([""] * (texCount - len(modes)))
        modes.extend(["MODULATE"] * (texCount - len(sources)))

        chains = {"diffuse": [], "specular": []}
        currentChain = None
        for i in range(0, texCount):
            if sources[i] == "DIFFUSE":
                currentChain = "diffuse"
                chains[currentChain] = [i] #Start new chain
            elif sources[i] == "SPECULAR":
                currentChain = "specular"
                chains[currentChain] = [i]
            else:
                if currentChain is not None:
                    chains[currentChain].append(i)
                else:
                    raise Exception() #Unsupported sequence

        for node in chains["diffuse"]:
            if modes[node] == "MODULATE":
                self.mapping["diffuse"] = node
            elif modes[node] == "DOTPRODUCT3":
                self.mapping["normal"] = node
            else:
                raise Exception() #Unsupported mode
        for node in chains["specular"]:
            if modes[node] == "MODULATE":
                self.mapping["specular"] = node
            else:
                raise Exception() #Unsupported mode


class X3dParser:
    def __init__(self):
        self.parser = expat.ParserCreate()
        self.parser.StartElementHandler = self.start
        self.parser.EndElementHandler = self.end
        self.parser.CharacterDataHandler = self.data
        self.entries = []
        self.scene = None
        self.current = None
        self.ignore = 0

    def feed(self, data):
        self.parser.Parse(data, 0)

    def close(self):
        self.parser.Parse("", 1) #End of data
        del self.parser #Get rid of circular references

    def start(self, tag, attributes):
#        level = self.current.level if self.current is not None else 0
#        debug("%sEnter tag %s, current %s" % (' ' * level, tag, self.current.__class__.__name__))
        
        if tag == "Scene":
            if self.scene is not None:
                debug("Error")
                raise Exception()
            else:
                self.scene = X3dScene()
                self.current = self.scene
        elif self.scene is not None:
            entry = None
            reused = False
            if "USE" in attributes.keys():
                entryName = attributes["USE"]
                defined = filter(lambda x: x.name == entryName, self.entries)
                if len(defined) > 0:
                    debug("%sReused entry %s" % (' ' * self.current.level, entryName))
                    entry = defined[0]
                    reused = True
                if entry is None:
                    debug("%sEntry %s not found" % (' ' * self.current.level, entryName))
            elif isinstance(self.current, (X3dScene, X3dTransform)):
                if tag in ("Transform", "Group", "Collision", "Switch"):                
                    entry = X3dTransform(self.current)
                elif tag == "Shape":
                    entry = X3dShape(self.current)
            elif isinstance(self.current, X3dShape):
                if tag == "Appearance":
                    entry = X3dAppearance(self.current)
                elif tag == "IndexedFaceSet":
                    entry = X3dGeometry(self.current)
            elif isinstance(self.current, X3dAppearance):
                if tag == "Material":
                    entry = X3dMaterial(self.current)
                elif tag == "ImageTexture":
                    entry = X3dTexture(self.current)
                elif tag == "MultiTexture":
                    entry = X3dMultiTexture(self.current)
            elif isinstance(self.current, X3dMultiTexture):
                if tag == "ImageTexture":
                    entry = X3dTexture(self.current)
            elif isinstance(self.current, X3dGeometry):
                if tag == "Coordinate":
                    entry = X3dGeoCoords(self.current)
                elif tag == "TextureCoordinate":
                    entry = X3dTexCoords(self.current)
            if entry is not None:
                if not reused:
                    entry.parseAttributes(attributes)
                self.current.ancestors.append(entry)
                if not reused:
                    self.current = entry
                    if entry.name != "":
                        self.entries.append(entry)
                else:
                    self.ignore += 1
            else:
                self.ignore += 1

    def end(self, tag):
#        level = self.current.level if self.current is not None else 0
#        debug("%sExit tag %s, current %s" % (' ' * level, tag, self.current.__class__.__name__))

        if self.ignore > 0:
            self.ignore -= 1
            return
        if self.current is not None and self.current.parent is not None:
            self.current = self.current.parent

    def data(self, data):
        pass
    
    def extract(self):
        return self.scene.extract() if self.scene is not None else []


def load(path):
    parser = X3dParser()
    xml = open(path, 'rb')
    content = xml.read()
    xml.close()
    parser.feed(content)
    parser.close()
    return parser.extract()

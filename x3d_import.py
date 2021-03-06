#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# x3d_import.py
# Copyright (C) 2015 xent
# Project is distributed under the terms of the GNU General Public License v3.0

import numpy
import re
from xml.parsers import expat

try:
    import model
except ImportError:
    from . import model

debugEnabled = False

def debug(text):
    if debugEnabled:
        print(text)


class X3dEntry:
    def __init__(self, parent=None):
        self.ident = None
        self.parent = parent
        self.name = ''
        self.ancestors = []
        self.level = self.parent.level + 1 if self.parent is not None else -1

    def demangled(self):
        return self.name

    def parse(self, attributes):
        pass

    def parseAttributes(self, attributes):
        if 'DEF' in attributes:
            self.name = attributes['DEF']
            debug('{:s}Entry name {:s}'.format('  ' * self.level, self.name))
        self.parse(attributes)


class X3dScene(X3dEntry):
    def __init__(self):
        super().__init__()
        self.transform = model.Transform()

    def extract(self):
        exportedMaterials, exportedMeshes = [], []

        def createMesh(geometry, appearance, name):
            # Create abstract mesh object
            mesh = model.Mesh(name=name)
            mesh.geoPolygons = geometry.geoPolygons
            mesh.texPolygons = geometry.texPolygons
            if appearance is not None:
                newMaterial = appearance.squash()
                materials = [mat for mat in exportedMaterials if mat == newMaterial]
                if len(materials) > 0:
                    debug('Squash: reused material {:s}'.format(materials[0].color.ident))
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
            polygons = [[translated[i] for i in poly] for poly in mesh.geoPolygons]

            debug('Reindex: mesh {:s}, {:d} polygons, from {:d} to {:d} vertices'.format(
                    mesh.ident, len(polygons), len(mesh.geoVertices), len(vertices)))
            mesh.geoPolygons = polygons
            mesh.geoVertices = vertices

        def squash(entry, transform, name=[]):
            if isinstance(entry, X3dTransform):
                parts = []
                for i in range(0, len(entry.ancestors)):
                    shape = entry.ancestors[i]
                    demangled = entry.demangled() if entry.demangled() != '' else entry.name
                    subname = name + [demangled] if demangled != '' else name
                    subname[-1] += '_' + str(i) if len(entry.ancestors) > 1 else ''
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
                        debug('Squash: reused mesh {:s}'.format(name[-1]))
                        # Create concrete shape
                        currentMesh = model.Mesh(parent=alreadyExported, name=name[0])
                        currentMesh.transform = transform
                        return [currentMesh]
                    else:
                        mesh = createMesh(geometry, appearance, name[-1])
                        if entry.parent is not None and entry.parent.name != '' and len(entry.parent.ancestors) > 1:
                            reindexMesh(mesh)
                        exportedMeshes.append(mesh)
                        # Create concrete shape
                        currentMesh = model.Mesh(parent=mesh, name=name[0])
                        currentMesh.transform = transform
                        return [currentMesh]
            return []
        entries = []
        [entries.extend(squash(x, self.transform)) for x in self.ancestors]
        return entries


class X3dTransform(X3dEntry):
    def __init__(self, parent):
        super().__init__(parent)
        self.transform = model.Transform()

    def parse(self, attributes):
        def getValues(string):
            return [float(x) for x in string.split(' ')]

        if 'rotation' in attributes:
            values = getValues(attributes['rotation'])
            vector, angle = model.normalize(values[0:3]), values[3]
            vector = model.normalize(vector)
            self.transform.rotate(vector, angle)
            debug('{:s}Rotation:    {:.3f} about [{:.3f}, {:.3f}, {:.3f}]'.format(
                    '  ' * (self.level + 1), values[3], *values[0:3]))
        if 'scale' in attributes:
            values = getValues(attributes['scale'])
            self.transform.scale(values)
            debug('{:s}Scale:       [{:.3f}, {:.3f}, {:.3f}]'.format(
                    '  ' * (self.level + 1), *values))
        if 'translation' in attributes:
            values = getValues(attributes['translation'])
            self.transform.translate(values)
            debug('{:s}Translation: [{:.3f}, {:.3f}, {:.3f}]'.format(
                    '  ' * (self.level + 1), *values))

    def demangled(self):
        # Demangle Blender names
        name = self.name.replace('OB_', '').replace('group_ME_', '')
        name = name.replace('_ifs', '').replace('_TRANSFORM', '')
        # Demangle own names
        name = name.replace('ME_', '')

        return name


class X3dShape(X3dEntry):
    def __init__(self, parent):
        super().__init__(parent)


class X3dGeometry(X3dEntry):
    POLYGON_PATTERN = re.compile(r'([ ,\\t\d]+)-1', re.I | re.S)

    def __init__(self, parent):
        super().__init__(parent)
        self.geoPolygons = []
        self.texPolygons = []
        self.smooth = False
        self.solid = True

    def parse(self, attributes):
        if 'solid' in attributes:
            self.solid = True if attributes['solid'] == 'true' else False
        if 'smooth' in attributes:
            self.smooth = True if attributes['smooth'] == 'true' else False

        def parsePolygons(string):
            chunks = []
            for entry in string.split('-1'):
                poly = entry.strip()
                if len(poly) > 0:
                    chunks.append(poly)
            polygons = []
            for poly in chunks:
                polygons.append([int(vertex) for vertex in poly.split(' ') if len(vertex) > 0])
            return polygons

        if 'coordIndex' in attributes:
            self.geoPolygons = parsePolygons(attributes['coordIndex'])
            debug('{:s}Found {:d} polygons'.format('  ' * self.level, len(self.geoPolygons)))
        if 'texCoordIndex' in attributes:
            self.texPolygons = parsePolygons(attributes['texCoordIndex'])
            debug('{:s}Found {:d} texture polygons'.format('  ' * self.level, len(self.texPolygons)))


class X3dCoords(X3dEntry):
    def __init__(self, parent, size):
        super().__init__(parent)
        self.size = size
        self.vertices = None

    def parse(self, attributes):
        if 'point' in attributes:
            points = [float(point) for point in attributes['point'].split(' ') if len(point) > 0]
            vertices = []
            [vertices.append(numpy.array(points[i * self.size:(i + 1) * self.size]))
                    for i in range(0, int(len(points) / self.size))]
            self.vertices = vertices
            debug('{:s}Found {:d} vertices, width {:d}'.format('  ' * self.level, len(self.vertices), self.size))


class X3dGeoCoords(X3dCoords):
    def __init__(self, parent):
        super().__init__(parent, 3)


class X3dTexCoords(X3dCoords):
    def __init__(self, parent):
        super().__init__(parent, 2)


class X3dAppearance(X3dEntry):
    def __init__(self, parent):
        super().__init__(parent)
        # Limited support for image textures: TextureTransform nodes unsupported

    def squash(self):
        material = model.Material()
        for entry in self.ancestors:
            if isinstance(entry, X3dMaterial):
                material.color = entry.color
            elif isinstance(entry, X3dTexture):
                material.diffuse = entry.texture
            elif isinstance(entry, X3dMultiTexture):
                if max(filter(lambda x: x is not None, entry.mapping.values())) >= len(entry.ancestors):
                    raise Exception() # Texture index is out of range
                if entry.mapping['diffuse'] is not None:
                    material.diffuse = entry.ancestors[entry.mapping['diffuse']].texture
                if entry.mapping['normal'] is not None:
                    material.normal = entry.ancestors[entry.mapping['normal']].texture
                if entry.mapping['specular'] is not None:
                    material.specular = entry.ancestors[entry.mapping['specular']].texture
        return material


class X3dMaterial(X3dEntry):
    IDENT = 0

    def __init__(self, parent):
        super().__init__(parent)
        self.color = model.Material.Color('DefaultX3dMaterial_{:d}'.format(X3dMaterial.IDENT))
        X3dMaterial.IDENT += 1

    def parse(self, attributes):
        def getValues(string):
            return numpy.array([float(x) for x in string.split(' ')])

        self.color.ident = self.demangled()
        if 'shininess' in attributes:
            self.color.shininess = float(attributes['shininess'])
        if 'transparency' in attributes:
            self.color.transparency = float(attributes['transparency'])
        if 'diffuseColor' in attributes:
            self.color.diffuse = getValues(attributes['diffuseColor'])
        if 'emissiveColor' in attributes:
            self.color.emissive = getValues(attributes['emissiveColor'])
        if 'specularColor' in attributes:
            self.color.specular = getValues(attributes['specularColor'])
        if 'ambientIntensity' in attributes:
            self.color.ambient = self.color.diffuse * float(attributes['ambientIntensity'])

        debug('{:s}Material properties:'.format('  ' * self.level))
        debug('{:s}Shininess:      {:.3f}'.format('  ' * (self.level + 1), self.color.shininess))
        debug('{:s}Transparency:   {:.3f}'.format('  ' * (self.level + 1), self.color.transparency))
        debug('{:s}Diffuse Color:  [{:.3f}, {:.3f}, {:.3f}]'.format('  ' * (self.level + 1), *self.color.diffuse))
        debug('{:s}Emissive Color: [{:.3f}, {:.3f}, {:.3f}]'.format('  ' * (self.level + 1), *self.color.emissive))
        debug('{:s}Specular Color: [{:.3f}, {:.3f}, {:.3f}]'.format('  ' * (self.level + 1), *self.color.specular))
        debug('{:s}Ambient Color:  [{:.3f}, {:.3f}, {:.3f}]'.format('  ' * (self.level + 1), *self.color.ambient))

    def demangled(self):
        # Demangle Blender names
        return self.name.replace('MA_', '')


class X3dTexture(X3dEntry):
    IDENT = 0

    def __init__(self, parent):
        super().__init__(parent)
        self.texture = model.Material.Texture('', 'DefaultX3dTexture_{:d}'.format(X3dTexture.IDENT))
        X3dTexture.IDENT += 1

    def parse(self, attributes):
        if 'url' in attributes:
            self.texture.path = [x.replace('\"', '') for x in attributes['url'].split(' ')]


class X3dMultiTexture(X3dEntry):
    def __init__(self, parent):
        super().__init__(parent)
        # Limited support for this type of node
        self.mapping = {'diffuse': None, 'normal': None, 'specular': None}

    def parse(self, attributes):
        modes, sources = [], []
        if 'source' in attributes:
            sources = [x.replace('\"', '') for x in attributes['source'].split(' ')]
        if 'mode' in attributes:
            modes = [x.replace('\"', '') for x in attributes['mode'].split(' ')]
        texCount = max(len(modes), len(sources))
        modes.extend([''] * (texCount - len(modes)))
        modes.extend(['MODULATE'] * (texCount - len(sources)))

        chains = {'diffuse': [], 'specular': []}
        currentChain = None
        for i in range(0, texCount):
            if sources[i] == 'DIFFUSE':
                currentChain = 'diffuse'
                chains[currentChain] = [i] # Start new chain
            elif sources[i] == 'SPECULAR':
                currentChain = 'specular'
                chains[currentChain] = [i]
            else:
                if currentChain is not None:
                    chains[currentChain].append(i)
                else:
                    raise Exception() # Unsupported sequence

        for node in chains['diffuse']:
            if modes[node] == 'MODULATE':
                self.mapping['diffuse'] = node
            elif modes[node] == 'DOTPRODUCT3':
                self.mapping['normal'] = node
            else:
                raise Exception() # Unsupported mode
        for node in chains['specular']:
            if modes[node] == 'MODULATE':
                self.mapping['specular'] = node
            else:
                raise Exception() # Unsupported mode


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
        self.parser.Parse('', 1) # End of data
        del self.parser # Get rid of circular references

    def start(self, tag, attributes):
        # level = self.current.level if self.current is not None else 0
        # debug('{:s}Enter tag {:s}, current {:s}.format('  ' * level, tag, self.current.__class__.__name__))

        if tag == 'Scene':
            if self.scene is not None:
                debug('Error')
                raise Exception()
            else:
                self.scene = X3dScene()
                self.current = self.scene
        elif self.scene is not None:
            entry = None
            reused = False
            if 'USE' in attributes:
                entryName = attributes['USE']
                defined = [entry for entry in self.entries if entry.name == entryName]
                if len(defined) > 0:
                    debug('{:s}Reused entry {:s}'.format('  ' * self.current.level, entryName))
                    entry = defined[0]
                    reused = True
                if entry is None:
                    debug('{:s}Entry {:s} not found'.format('  ' * self.current.level, entryName))
            elif isinstance(self.current, (X3dScene, X3dTransform)):
                if tag in ('Transform', 'Group', 'Collision', 'Switch'):
                    entry = X3dTransform(self.current)
                elif tag == 'Shape':
                    entry = X3dShape(self.current)
            elif isinstance(self.current, X3dShape):
                if tag == 'Appearance':
                    entry = X3dAppearance(self.current)
                elif tag == 'IndexedFaceSet':
                    entry = X3dGeometry(self.current)
            elif isinstance(self.current, X3dAppearance):
                if tag == 'Material':
                    entry = X3dMaterial(self.current)
                elif tag == 'ImageTexture':
                    entry = X3dTexture(self.current)
                elif tag == 'MultiTexture':
                    entry = X3dMultiTexture(self.current)
            elif isinstance(self.current, X3dMultiTexture):
                if tag == 'ImageTexture':
                    entry = X3dTexture(self.current)
            elif isinstance(self.current, X3dGeometry):
                if tag == 'Coordinate':
                    entry = X3dGeoCoords(self.current)
                elif tag == 'TextureCoordinate':
                    entry = X3dTexCoords(self.current)
            if entry is not None:
                if not reused:
                    entry.parseAttributes(attributes)
                self.current.ancestors.append(entry)
                if not reused:
                    self.current = entry
                    if entry.name != '':
                        self.entries.append(entry)
                else:
                    self.ignore += 1
            else:
                self.ignore += 1

    def end(self, tag):
        # level = self.current.level if self.current is not None else 0
        # debug('{:s}Exit tag {:s}, current {:s}'.format('  ' * level, tag, self.current.__class__.__name__))

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

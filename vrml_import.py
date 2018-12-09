#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# vrml_import.py
# Copyright (C) 2015 xent
# Project is distributed under the terms of the GNU General Public License v3.0

import functools
import numpy
import os
import re

try:
    import model
except ImportError:
    from . import model

debugEnabled = False

def debug(text):
    if debugEnabled:
        print(text)

def skipChunk(stream):
    balance = 1
    while True:
        data = stream.readline().decode('utf-8')
        if len(data) == 0:
            break
        for i in range(0, len(data)):
            if data[i] in ('[', '{'):
                balance += 1
            if data[i] in (']', '}'):
                balance -= 1
            if balance == 0:
                return len(data) - i - 2
    return 0

def calcBalance(string, delta=None, openset=('[', '{'), closeset=(']', '}')):
    balance, offset = 0, 0
    update = False

    for i in range(0, len(string)):
        if string[i] in openset:
            balance += 1
            update = False
        if string[i] in closeset:
            balance -= 1
            update = True
        if update and delta is not None and balance >= delta:
            offset = len(string) - i - 1
            update = False
    return (balance, offset)


class VrmlEntry:
    DEF_PATTERN = re.compile(r'([\w]*?)\s*([\w\.\-]*?)\s*(\w+)\s*{', re.I | re.S)
    IDENT = 0

    def __init__(self, parent=None):
        self.ident = None
        self.parent = parent
        self.name = ''
        self.objects = []
        self.level = self.parent.level + 2 if self.parent is not None else 0

    def chain(self, entryType):
        if isinstance(self, (VrmlScene, VrmlTransform, vrmlInline)):
            # Collision and Switch nodes functionality unimplemented
            if entryType in ('Transform', 'Group', 'Collision', 'Switch'):
                return VrmlTransform(self)
            elif entryType == 'Inline':
                return vrmlInline(self)
            elif entryType == 'Shape':
                return VrmlShape(self)
            else:
                raise Exception()
        elif isinstance(self, VrmlShape):
            if entryType == 'Appearance':
                return VrmlAppearance(self)
            elif entryType == 'IndexedFaceSet':
                return VrmlGeometry(self)
            else:
                raise Exception()
        elif isinstance(self, VrmlAppearance):
            if entryType == 'Material':
                return VrmlMaterial(self)
            elif entryType == 'ImageTexture':
                return VrmlTexture(self)
            else:
                raise Exception()
        elif isinstance(self, VrmlGeometry):
            if entryType == 'Coordinate':
                return VrmlGeoCoords(self)
            elif entryType == 'TextureCoordinate':
                return VrmlTexCoords(self)
            else:
                raise Exception()
        else:
            raise Exception()

    def readStream(self, stream):
        delta, offset, balance = 0, 0, 0
        # Highest level
        while True:
            data = stream.readline().decode('utf-8')
            if len(data) == 0:
                break
            regexp = VrmlEntry.DEF_PATTERN.search(data)
            if regexp is not None:
                delta, offset = calcBalance(data[:regexp.start()], -1, ('{'), ('}'))
                balance += delta
                initialPos = stream.tell()
                self.read(stream, data[:regexp.start()])
                if initialPos != stream.tell():
                    print('{:s}Read error'.format(' ' * self.level))
                    break
                if balance < 0:
                    debug('{:s}Wrong balance: {:d}'.format(' ' * self.level, balance))
                    stream.seek(-(len(data) - regexp.start() + offset), os.SEEK_CUR)
                    break
                stream.seek(-(len(data) - regexp.end()), os.SEEK_CUR)
                entry = None
                debug('{:s}Entry: "{:s}" "{:s}" "{:s}" Balance: {:d}'.format(
                        ' ' * self.level, regexp.group(1), regexp.group(2), regexp.group(3), balance))

                try:
                    entry = self.chain(regexp.group(3))
                except:
                    debug('{:s}Unsupported chunk sequence: {:s}->{:s}'.format(
                            ' ' * self.level, self.__class__.__name__, regexp.group(3)))
                    offset = skipChunk(stream)
                    stream.seek(-offset, os.SEEK_CUR)

                if entry:
                    if regexp.group(1) == 'DEF' and len(regexp.group(2)) > 0:
                        entry.name = regexp.group(2)
                    entry.readStream(stream)
                    ptr = self
                    inline = None
                    while not isinstance(ptr, VrmlScene):
                        if inline is None and isinstance(ptr, vrmlInline):
                            inline = ptr
                        ptr = ptr.parent

                    duplicate = False
                    # Search for duplicates
                    for current in ptr.entries:
                        if entry == current:
                            debug('{:s}Not unique, using entry with id: {:d}'.format(' ' * self.level, current.id))
                            entry = current
                            duplicate = True
                            break

                    self.objects.append(entry)
                    if inline:
                        inline.entries.append(entry)
                    if not duplicate:
                        entry.id = VrmlEntry.IDENT
                        VrmlEntry.IDENT += 1
                        ptr.entries.append(entry)
            else:
                delta, offset = calcBalance(data, -(balance + 1), ('{'), ('}'))
                balance += delta
                initialPos = stream.tell()
                self.read(stream, data)
                using = re.search('USE\s+([\w\.\-]+)', data, re.I | re.S)
                if using is not None and using.start() < len(data) - offset:
                    debug('{:s}Using entry {:s}'.format(' ' * self.level, using.group(1)))
                    ptr = self
                    while not isinstance(ptr, (vrmlInline, VrmlScene)):
                        ptr = ptr.parent
                    self.objects.extend(filter(lambda x: x.name == using.group(1), ptr.entries))
                if balance < 0:
                    debug('{:s}Balance mismatch: {:d}'.format(' ' * self.level, balance))
                    self.finalize()
                    if initialPos == stream.tell():
                        stream.seek(-offset, os.SEEK_CUR)
                    break

    def demangled(self):
        return self.name

    def finalize(self):
        pass

    def read(self, stream, string):
        pass


class VrmlScene(VrmlEntry):
    def __init__(self, path):
        super().__init__()
        self.entries = []
        self.transform = model.Transform()

        inputFile = open(path, 'rb')
        oldDir = os.getcwd()
        if len(os.path.dirname(path)) > 0:
            os.chdir(os.path.dirname(path))
        self.readStream(inputFile)
        os.chdir(oldDir)
        inputFile.close()

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
            mesh.smooth = geometry.smooth
            mesh.solid = geometry.solid
            for subentry in geometry.objects:
                if isinstance(subentry, VrmlGeoCoords):
                    mesh.geoVertices = subentry.vertices
                elif isinstance(subentry, VrmlTexCoords):
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
            if isinstance(entry, VrmlTransform):
                parts = []
                for i in range(0, len(entry.objects)):
                    shape = entry.objects[i]
                    demangled = entry.demangled() if entry.demangled() != '' else entry.name
                    subname = name + [demangled] if demangled != '' else name
                    subname[-1] += '_' + str(i) if len(entry.objects) > 1 else ''
                    parts.extend(squash(shape, transform * entry.transform, subname))
                return parts
            elif isinstance(entry, VrmlShape):
                appearance, geometry = None, None
                for subentry in entry.objects:
                    if isinstance(subentry, VrmlAppearance):
                        appearance = subentry
                    if isinstance(subentry, VrmlGeometry):
                        geometry = subentry
                if geometry is not None:
                    alreadyExported = [mesh for mesh in exportedMeshes if mesh.ident == name[-1]]
                    if len(alreadyExported) > 0:
                        debug('Squash: reused mesh {:s}'.format(name[-1]))
                        # Create concrete shape
                        currentMesh = model.Mesh(parent=alreadyExported[0], name=name[0])
                        currentMesh.transform = transform
                        return [currentMesh]
                    else:
                        mesh = createMesh(geometry, appearance, name[-1])
                        if entry.parent is not None and entry.parent.name != '' and len(entry.parent.objects) > 1:
                            reindexMesh(mesh)
                        exportedMeshes.append(mesh)
                        # Create concrete shape
                        currentMesh = model.Mesh(parent=mesh, name=name[0])
                        currentMesh.transform = transform
                        return [currentMesh]
            return []
        entries = []
        [entries.extend(squash(x, self.transform)) for x in self.objects]
        return entries


class VrmlTransform(VrmlEntry):
    def __init__(self, parent):
        super().__init__(parent)
        self.transform = model.Transform()

    def read(self, stream, string):
        key = '([+e\d\-\.]+)'
        result = re.search('translation\s+' + '\s+'.join([key] * 3), string, re.I | re.S)
        if result is not None:
            self.transform.translate([float(result.group(x)) for x in range(1, 4)])
        result = re.search('rotation\s+' + '\s+'.join([key] * 4), string, re.I | re.S)
        if result is not None:
            values = [float(result.group(x)) for x in range(1, 4)]
            vector, angle = model.normalize(values), float(result.group(4))
            vector = model.normalize(vector)
            self.transform.rotate(vector, angle)
        result = re.search('scale\s+' + '\s+'.join([key] * 3), string, re.I | re.S)
        if result is not None:
            self.transform.scale([float(result.group(x)) for x in range(1, 4)])

    def demangled(self):
        # Demangle Blender names
        name = self.name.replace('OB_', '').replace('group_ME_', '').replace('_ifs_TRANSFORM', '')
        # Demangle own names
        name = name.replace('ME_', '')

        return name


class vrmlInline(VrmlTransform):
    def __init__(self, parent):
        super().__init__(parent)
        self.entries = []

    def read(self, stream, string):
        urlSearch = re.search('url\s+\'([\w\-\._\/]+)\'', string, re.S)
        if urlSearch is not None:
            oldDir = os.getcwd()
            if os.path.isfile(urlSearch.group(1)):
                debug('{:s}Loading file: {:s}'.format(' ' * self.level, urlSearch.group(1)))
                inputFile = open(urlSearch.group(1), 'r')
                if len(os.path.dirname(urlSearch.group(1))) > 0:
                    os.chdir(os.path.dirname(urlSearch.group(1)))
                self.readStream(inputFile)
                inputFile.close()
            else:
                print('Inline file not found: {:s}'.format(urlSearch.group(1)))
            os.chdir(oldDir)


class VrmlShape(VrmlEntry):
    def __init__(self, parent):
        super().__init__(parent)


class VrmlGeometry(VrmlEntry):
    POLYGON_PATTERN = re.compile(r'([ ,\\t\d]+)-1', re.I | re.S)

    def __init__(self, parent):
        super().__init__(parent)
        self.geoPolygons = []
        self.texPolygons = []
        self.smooth = False
        self.solid = False

    def read(self, stream, string):
        initialPos = stream.tell()

        self.solid = re.search('solid\s+TRUE', string, re.S) is not None
        self.smooth = re.search('smooth\s+TRUE', string, re.S) is not None

        searchType = re.search('(\w+)Index\s*\[', string, re.S)
        if searchType is None:
            return
        coordinates = searchType.group(1) == 'coord'

        polygons = []
        delta, offset, balance = 0, 0, 0
        data = string
        position = searchType.end()

        debug('{:s}Start {:s} polygons read'.format(' ' * self.level, 'coordinate' if coordinates else 'texture'))
        while True:
            while True:
                regexp = VrmlGeometry.POLYGON_PATTERN.search(data, position)
                if regexp:
                    delta, offset = calcBalance(data[position:regexp.start()], -1, (), (']'))
                    balance += delta
                    offset = len(data) - regexp.start() + offset
                    if balance != 0:
                        debug('{:s}Wrong balance: {:d}, offset: {:d}'.format(' ' * self.level, balance, offset))
                        break

                    vertexString = regexp.group(1).replace(',', '').replace('\t', '')
                    vertices = [int(x) for x in vertexString.split(' ') if len(x) > 0]

                    try:
                        polygons.extend(model.Mesh.tesselate(vertices))
                    except:
                        debug('{:s}Wrong polygon vertex count: {:d}'.format(' ' * self.level, len(vertices)))
                        break
                    position = regexp.end()
                else:
                    delta, offset = calcBalance(data, None, (), (']'))
                    balance += delta
                    offset = len(data)
                    break
            if balance != 0:
                if initialPos != stream.tell():
                    stream.seek(-offset, os.SEEK_CUR)
                debug('{:s}Balance mismatch: {:d}, offset: {:d}'.format(' ' * self.level, balance, offset))
                break
            data = stream.readline().decode('utf-8')
            if len(data) == 0:
                break
            position = 0
        if coordinates:
            self.geoPolygons = polygons
            debug('{:s}Read poly done, {:d} polygons'.format(' ' * self.level, len(self.geoPolygons)))
        else:
            self.texPolygons = polygons
            debug('{:s}Read UV poly done, {:d} polygons'.format(' ' * self.level, len(self.texPolygons)))


class VrmlCoords(VrmlEntry):
    def __init__(self, parent, size):
        super().__init__(parent)
        self.size = size
        self.vertices = None
        self.pattern = re.compile(r'[ ,\\t]+'.join([r'([+e\d\-\.]+)'] * size), re.I | re.S)

    def read(self, stream, string):
        initialPos = stream.tell()
        indexSearch = re.search('point\s*\[', string, re.S)

        if indexSearch is not None:
            debug('{:s}Start vertex read, width: {:d}'.format(' ' * self.level, self.size))

            self.vertices = []
            delta, offset, balance = 0, 0, 0
            data = string
            vPos = indexSearch.end()
            while True:
                while True:
                    regexp = self.pattern.search(data, vPos)
                    if regexp:
                        delta, offset = calcBalance(data[vPos:regexp.start()], -1, (), ('}'))
                        balance += delta
                        offset = len(data) - regexp.start() + offset
                        if initialPos != stream.tell():
                            offset += 1
                        if balance != 0:
                            debug('{:s}Wrong balance: {:d}, offset: {:d}'.format(' ' * self.level, balance, offset))
                            break
                        vertices = [float(regexp.group(i + 1)) for i in range(0, self.size)]
                        self.vertices.append(numpy.array(vertices))
                        vPos = regexp.end()
                    else:
                        delta, offset = calcBalance(data[vPos:], -1, (), ('}'))
                        balance += delta
                        if initialPos != stream.tell():
                            offset += 1
                        break
                if balance != 0:
                    if initialPos != stream.tell():
                        stream.seek(-offset, os.SEEK_CUR)
                    debug('{:s}Balance mismatch: {:d}, offset: {:d}'.format(' ' * self.level, balance, offset))
                    break
                data = stream.readline().decode('utf-8')
                if len(data) == 0:
                    break
                vPos = 0
            debug('{:s}End vertex read, count: {:d}'.format(' ' * self.level, len(self.vertices)))


class VrmlGeoCoords(VrmlCoords):
    def __init__(self, parent):
        super().__init__(parent, 3)


class VrmlTexCoords(VrmlCoords):
    def __init__(self, parent):
        super().__init__(parent, 2)


class VrmlAppearance(VrmlEntry):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.diffuse = None
        self.normal = None
        self.specular = None

    def __eq__(self, other):
        if not isinstance(other, VrmlAppearance):
            return False
        if len(self.objects) != len(other.objects) or len(self.objects) == 0:
            return False
        return functools.reduce(lambda a, b: a and b, map(lambda c: c in other.objects, self.objects))

    def __ne__(self, other):
        return not self == other

    def squash(self):
        material = model.Material()
        for entry in self.objects:
            if isinstance(entry, VrmlMaterial):
                material.color = entry.color
            elif isinstance(entry, VrmlTexture):
                material.diffuse = entry.texture
        return material


class VrmlMaterial(VrmlEntry):
    IDENT = 0
    PATTERNS = [(1, 'shininess'), (1, 'transparency'), (1, 'ambientIntensity'),
            (3, 'diffuseColor'), (3, 'emissiveColor'), (3, 'specularColor')]

    def __init__(self, parent):
        super().__init__(parent)
        self.color = model.Material.Color('DefaultVrmlMaterial_{:d}'.format(VrmlMaterial.IDENT))
        self.values = {}
        VrmlMaterial.IDENT += 1

    def read(self, stream, string):
        key = '([+e\d\-\.]+)'
        for pattern in VrmlMaterial.PATTERNS:
            result = re.search(pattern[1] + '\s+' + '\s'.join([key] * pattern[0]), string, re.I | re.S)
            if result is not None:
                values = [float(result.group(x)) for x in range(1, pattern[0] + 1)]
                debug('{:s}Material attribute {:s} found'.format(' ' * self.level, pattern[1]))
                self.values[pattern[1]] = values[0] if len(values) == 1 else numpy.array(values)

    def demangled(self):
        # Demangle Blender names
        return self.name.replace('MA_', '')

    def finalize(self):
        self.color.ident = self.demangled()
        if 'shininess' in self.values:
            self.color.shininess = self.values['shininess']
        if 'transparency' in self.values:
            self.color.transparency = self.values['transparency']
        if 'diffuseColor' in self.values:
            self.color.diffuse = self.values['diffuseColor']
        if 'emissiveColor' in self.values:
            self.color.emissive = self.values['emissiveColor']
        if 'specularColor' in self.values:
            self.color.specular = self.values['specularColor']
        if 'ambientIntensity' in self.values:
            self.color.ambient = self.color.diffuse * self.values['ambientIntensity']

        debug('{:s}Material properties:'.format(' ' * self.level))
        debug('{:s}Shininess:      {:.3f}'.format(' ' * (self.level + 1), self.color.shininess))
        debug('{:s}Transparency:   {:.3f}'.format(' ' * (self.level + 1), self.color.transparency))
        debug('{:s}Diffuse Color:  {:.3f}, {:.3f}, {:.3f}'.format(' ' * (self.level + 1), *self.color.diffuse))
        debug('{:s}Emissive Color: {:.3f}, {:.3f}, {:.3f}'.format(' ' * (self.level + 1), *self.color.emissive))
        debug('{:s}Specular Color: {:.3f}, {:.3f}, {:.3f}'.format(' ' * (self.level + 1), *self.color.specular))
        debug('{:s}Ambient Color:  {:.3f}, {:.3f}, {:.3f}'.format(' ' * (self.level + 1), *self.color.ambient))

    def __eq__(self, other):
        if not isinstance(other, VrmlMaterial):
            return False
        return self.color == other.color

    def __ne__(self, other):
        return not self == other


class VrmlTexture(VrmlEntry):
    IDENT = 0

    def __init__(self, parent):
        super().__init__(parent)
        self.texture = model.Material.Texture('', 'DefaultVrmlTexture_{:d}'.format(VrmlTexture.IDENT))
        VrmlTexture.IDENT += 1

    def read(self, stream, string):
        tmp = re.search('url\s+\'([\w\-\.:\/]+)\'', string, re.I | re.S)
        if tmp is not None:
            path = (tmp.group(1), os.getcwd() + '/' + tmp.group(1))
            self.texture.path = path
            self.texture.ident = self.name

    def __eq__(self, other):
        if not isinstance(other, VrmlTexture):
            return False
        return self.texture == other.texture

    def __ne__(self, other):
        return not self == other


def load(path):
    scene = VrmlScene(path)
    return scene.extract()

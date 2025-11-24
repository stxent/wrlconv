#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# vrml_import.py
# Copyright (C) 2015 xent
# Project is distributed under the terms of the GNU General Public License v3.0

import functools
import os
import re
import numpy

try:
    import model
except ImportError:
    from . import model

DEBUG_ENABLED = False

def debug(text):
    if DEBUG_ENABLED:
        print(text)

def skip_chunk(stream):
    balance = 1
    while True:
        data = stream.readline().decode('utf-8')
        if not data:
            break
        for i, value in enumerate(data):
            if value in ('[', '{'):
                balance += 1
            if value in (']', '}'):
                balance -= 1
            if balance == 0:
                return len(data) - i - 2
    return 0

def calc_balance(string, delta=None, openset=('[', '{'), closeset=(']', '}')):
    balance, offset = 0, 0
    update = False

    for i, value in enumerate(string):
        if value in openset:
            balance += 1
            update = False
        if value in closeset:
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
        self.level_value = self.parent.level_value + 2 if self.parent is not None else 0

    def level(self, increment=0):
        return ' ' * (self.level_value + increment)

    def chain(self, entry_type):
        if isinstance(self, (VrmlScene, VrmlTransform, VrmlInline)):
            # Collision and Switch nodes functionality unimplemented
            if entry_type in ('Transform', 'Group', 'Collision', 'Switch'):
                return VrmlTransform(self)
            if entry_type == 'Inline':
                return VrmlInline(self)
            if entry_type == 'Shape':
                return VrmlShape(self)
            raise TypeError()
        if isinstance(self, VrmlShape):
            if entry_type == 'Appearance':
                return VrmlAppearance(self)
            if entry_type == 'IndexedFaceSet':
                return VrmlGeometry(self)
            raise TypeError()
        if isinstance(self, VrmlAppearance):
            if entry_type == 'Material':
                return VrmlMaterial(self)
            if entry_type == 'ImageTexture':
                return VrmlTexture(self)
            raise TypeError()
        if isinstance(self, VrmlGeometry):
            if entry_type == 'Coordinate':
                return VrmlGeoCoords(self)
            if entry_type == 'TextureCoordinate':
                return VrmlTexCoords(self)
            raise TypeError()
        raise TypeError()

    def read_stream(self, stream):
        delta, offset, balance = 0, 0, 0
        # Highest level
        while True:
            data = stream.readline().decode('utf-8')
            if not data:
                break
            regexp = VrmlEntry.DEF_PATTERN.search(data)
            if regexp is not None:
                delta, offset = calc_balance(data[:regexp.start()], -1, ('{'), ('}'))
                balance += delta
                initial_pos = stream.tell()
                self.read(stream, data[:regexp.start()])
                if initial_pos != stream.tell():
                    print('{:s}Read error'.format(self.level()))
                    break
                if balance < 0:
                    debug('{:s}Wrong balance: {:d}'.format(self.level(), balance))
                    stream.seek(-(len(data) - regexp.start() + offset), os.SEEK_CUR)
                    break
                stream.seek(-(len(data) - regexp.end()), os.SEEK_CUR)
                entry = None
                debug('{:s}Entry: "{:s}" "{:s}" "{:s}" Balance: {:d}'.format(
                    self.level(), regexp.group(1), regexp.group(2), regexp.group(3), balance))

                try:
                    entry = self.chain(regexp.group(3))
                except TypeError:
                    debug('{:s}Unsupported chunk sequence: {:s}->{:s}'.format(
                        self.level(), self.__class__.__name__, regexp.group(3)))
                    offset = skip_chunk(stream)
                    stream.seek(-offset, os.SEEK_CUR)

                if entry:
                    if regexp.group(1) == 'DEF' and regexp.group(2):
                        entry.name = regexp.group(2)
                    entry.read_stream(stream)
                    ptr = self
                    inline = None
                    while not isinstance(ptr, VrmlScene):
                        if inline is None and isinstance(ptr, VrmlInline):
                            inline = ptr
                        ptr = ptr.parent

                    duplicate = False
                    # Search for duplicates
                    for current in ptr.entries:
                        if entry == current:
                            debug('{:s}Not unique, using entry with id: {:d}'.format(
                                self.level(), current.ident))
                            entry = current
                            duplicate = True
                            break

                    self.objects.append(entry)
                    if inline:
                        inline.objects.append(entry)
                    if not duplicate:
                        entry.ident = VrmlEntry.IDENT
                        VrmlEntry.IDENT += 1
                        ptr.entries.append(entry)
            else:
                delta, offset = calc_balance(data, -(balance + 1), ('{'), ('}'))
                balance += delta
                initial_pos = stream.tell()
                self.read(stream, data)
                using = re.search(r'USE\s+([\w\.\-]+)', data, re.I | re.S)
                if using is not None and using.start() < len(data) - offset:
                    debug('{:s}Using entry {:s}'.format(self.level(), using.group(1)))
                    ptr = self
                    while not isinstance(ptr, (VrmlInline, VrmlScene)):
                        ptr = ptr.parent
                    self.objects.extend(filter(lambda x: x.name == using.group(1), ptr.entries))
                if balance < 0:
                    debug('{:s}Balance mismatch: {:d}'.format(self.level(), balance))
                    self.finalize()
                    if initial_pos == stream.tell():
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

        with open(path, 'rb') as input_file:
            old_dir = os.getcwd()
            if os.path.dirname(path):
                os.chdir(os.path.dirname(path))
            self.read_stream(input_file)
            os.chdir(old_dir)

    def extract(self):
        exported_materials, exported_meshes = [], []

        def create_mesh(geometry, appearance, name):
            # Create abstract mesh object
            mesh = model.Mesh(name=name)
            mesh.geo_polygons = geometry.geo_polygons
            mesh.tex_polygons = geometry.tex_polygons
            if appearance is not None:
                new_material = appearance.squash()
                materials = [mat for mat in exported_materials if mat == new_material]
                if materials:
                    debug('Squash: reused material {:s}'.format(materials[0].color.ident))
                    mesh.appearance().material = materials[0]
                else:
                    mesh.appearance().material = new_material
                    exported_materials.append(new_material)
            mesh.smooth = geometry.smooth
            mesh.solid = geometry.solid
            for subentry in geometry.objects:
                if isinstance(subentry, VrmlGeoCoords):
                    mesh.geo_vertices = subentry.vertices
                elif isinstance(subentry, VrmlTexCoords):
                    mesh.tex_vertices = subentry.vertices
            return mesh

        def reindex_mesh(mesh):
            used = []
            for poly in mesh.geo_polygons:
                used.extend([i for i in poly if i not in used])
            used.sort()

            vertices = [mesh.geo_vertices[i] for i in used]
            translated = dict(zip(used, range(0, len(vertices))))
            polygons = [[translated[i] for i in poly] for poly in mesh.geo_polygons]

            debug('Reindex: mesh {:s}, {:d} polygons, from {:d} to {:d} vertices'.format(
                mesh.ident, len(polygons), len(mesh.geo_vertices), len(vertices)))
            mesh.geo_polygons = polygons
            mesh.geo_vertices = vertices

        def squash(entry, transform, name=None):
            if name is None:
                name = []

            if isinstance(entry, VrmlTransform):
                parts = []
                for i, shape in enumerate(entry.objects):
                    demangled = entry.demangled() if entry.demangled() != '' else entry.name
                    subname = name + [demangled] if demangled != '' else name
                    subname[-1] += '_' + str(i) if len(entry.objects) > 1 else ''
                    parts.extend(squash(shape, transform * entry.transform, subname))
                return parts
            if isinstance(entry, VrmlShape):
                appearance, geometry = None, None
                for subentry in entry.objects:
                    if isinstance(subentry, VrmlAppearance):
                        appearance = subentry
                    if isinstance(subentry, VrmlGeometry):
                        geometry = subentry
                if geometry is not None:
                    already_exported = [mesh for mesh in exported_meshes if mesh.ident == name[-1]]
                    if already_exported:
                        debug('Squash: reused mesh {:s}'.format(name[-1]))
                        # Create concrete shape
                        current_mesh = model.Mesh(parent=already_exported[0], name=name[0])
                        current_mesh.transform = transform
                        return [current_mesh]

                    mesh = create_mesh(geometry, appearance, name[-1])
                    parent = entry.parent
                    if parent is not None and parent.name != '' and parent.objects:
                        reindex_mesh(mesh)
                    exported_meshes.append(mesh)
                    # Create concrete shape
                    current_mesh = model.Mesh(parent=mesh, name=name[0])
                    current_mesh.transform = transform
                    return [current_mesh]
            return []
        entries = []
        for entry in self.objects:
            entries.extend(squash(entry, self.transform))
        return entries


class VrmlTransform(VrmlEntry):
    def __init__(self, parent):
        super().__init__(parent)
        self.transform = model.Transform()

    def read(self, stream, string):
        key = r'([+e\d\-\.]+)'
        result = re.search(r'translation\s+' + r'\s+'.join([key] * 3), string, re.I | re.S)
        if result is not None:
            self.transform.translate([float(result.group(x)) for x in range(1, 4)])
        result = re.search(r'rotation\s+' + r'\s+'.join([key] * 4), string, re.I | re.S)
        if result is not None:
            values = [float(result.group(x)) for x in range(1, 4)]
            vector, angle = model.normalize(values), float(result.group(4))
            vector = model.normalize(vector)
            self.transform.rotate(vector, angle)
        result = re.search(r'scale\s+' + r'\s+'.join([key] * 3), string, re.I | re.S)
        if result is not None:
            self.transform.scale([float(result.group(x)) for x in range(1, 4)])

    def demangled(self):
        # Demangle Blender names
        name = self.name.replace('OB_', '').replace('group_ME_', '').replace('_ifs_TRANSFORM', '')
        # Demangle own names
        name = name.replace('ME_', '')

        return name


class VrmlInline(VrmlTransform):
    def __init__(self, parent):
        super().__init__(parent)
        self.entries = []

    def read(self, stream, string):
        url_search = re.search(r'url\s+\'([\w\-\._\/]+)\'', string, re.S)
        if url_search is not None:
            old_dir = os.getcwd()
            if os.path.isfile(url_search.group(1)):
                debug('{:s}Loading file: {:s}'.format(self.level(), url_search.group(1)))
                with open(url_search.group(1), 'rb') as input_file:
                    if os.path.dirname(url_search.group(1)):
                        os.chdir(os.path.dirname(url_search.group(1)))
                    self.read_stream(input_file)
            else:
                print('Inline file not found: {:s}'.format(url_search.group(1)))
            os.chdir(old_dir)


class VrmlShape(VrmlEntry):
    def __init__(self, parent):
        super().__init__(parent)


class VrmlGeometry(VrmlEntry):
    POLYGON_PATTERN = re.compile(r'([ ,\\t\d]+)-1', re.I | re.S)

    def __init__(self, parent):
        super().__init__(parent)
        self.geo_polygons = []
        self.tex_polygons = []
        self.smooth = False
        self.solid = False

    def read(self, stream, string):
        initial_pos = stream.tell()

        self.solid = re.search(r'solid\s+TRUE', string, re.S) is not None
        self.smooth = re.search(r'smooth\s+TRUE', string, re.S) is not None

        search_type = re.search(r'(\w+)Index\s*\[', string, re.S)
        if search_type is None:
            return
        coordinates = search_type.group(1) == 'coord'

        polygons = []
        delta, offset, balance = 0, 0, 0
        data = string
        position = search_type.end()

        debug('{:s}Start {:s} polygons read'.format(
            self.level(), 'coordinate' if coordinates else 'texture'))
        while True:
            while True:
                regexp = VrmlGeometry.POLYGON_PATTERN.search(data, position)
                if regexp:
                    delta, offset = calc_balance(data[position:regexp.start()], -1, (), (']'))
                    balance += delta
                    offset = len(data) - regexp.start() + offset
                    if balance != 0:
                        debug('{:s}Wrong balance: {:d}, offset: {:d}'.format(
                            self.level(), balance, offset))
                        break

                    vertex_string = regexp.group(1).replace(',', '').replace('\t', '')
                    vertices = [int(x) for x in vertex_string.split(' ') if x]

                    try:
                        polygons.extend(model.Mesh.triangulate(vertices))
                    except ValueError:
                        debug('{:s}Wrong polygon vertex count: {:d}'.format(
                            self.level(), len(vertices)))
                        break
                    position = regexp.end()
                else:
                    delta, offset = calc_balance(data, None, (), (']'))
                    balance += delta
                    offset = len(data)
                    break
            if balance != 0:
                if initial_pos != stream.tell():
                    stream.seek(-offset, os.SEEK_CUR)
                debug('{:s}Balance mismatch: {:d}, offset: {:d}'.format(
                    self.level(), balance, offset))
                break
            data = stream.readline().decode('utf-8')
            if not data:
                break
            position = 0
        if coordinates:
            self.geo_polygons = polygons
            debug('{:s}Read poly done, {:d} polygons'.format(
                self.level(), len(self.geo_polygons)))
        else:
            self.tex_polygons = polygons
            debug('{:s}Read UV poly done, {:d} polygons'.format(
                self.level(), len(self.tex_polygons)))


class VrmlCoords(VrmlEntry):
    def __init__(self, parent, size):
        super().__init__(parent)
        self.size = size
        self.vertices = None
        self.pattern = re.compile(r'[ ,\\t]+'.join([r'([+e\d\-\.]+)'] * size), re.I | re.S)

    def read(self, stream, string):
        initial_pos = stream.tell()
        index_search = re.search(r'point\s*\[', string, re.S)

        if index_search is not None:
            debug('{:s}Start vertex read, width: {:d}'.format(self.level(), self.size))

            self.vertices = []
            delta, offset, balance = 0, 0, 0
            data = string
            v_pos = index_search.end()
            while True:
                while True:
                    regexp = self.pattern.search(data, v_pos)
                    if regexp:
                        delta, offset = calc_balance(data[v_pos:regexp.start()], -1, (), ('}'))
                        balance += delta
                        offset = len(data) - regexp.start() + offset
                        if initial_pos != stream.tell():
                            offset += 1
                        if balance != 0:
                            debug('{:s}Wrong balance: {:d}, offset: {:d}'.format(
                                self.level(), balance, offset))
                            break
                        vertices = [float(regexp.group(i + 1)) for i in range(0, self.size)]
                        self.vertices.append(numpy.array(vertices))
                        v_pos = regexp.end()
                    else:
                        delta, offset = calc_balance(data[v_pos:], -1, (), ('}'))
                        balance += delta
                        if initial_pos != stream.tell():
                            offset += 1
                        break
                if balance != 0:
                    if initial_pos != stream.tell():
                        stream.seek(-offset, os.SEEK_CUR)
                    debug('{:s}Balance mismatch: {:d}, offset: {:d}'.format(
                        self.level(), balance, offset))
                    break
                data = stream.readline().decode('utf-8')
                if not data:
                    break
                v_pos = 0
            debug('{:s}End vertex read, count: {:d}'.format(self.level(), len(self.vertices)))


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
        if len(self.objects) != len(other.objects) or not self.objects:
            return False
        intersection = map(lambda c: c in other.objects, self.objects)
        return functools.reduce(lambda a, b: a and b, intersection)

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
    PATTERNS = [
        (1, 'shininess'),
        (1, 'transparency'),
        (1, 'ambientIntensity'),
        (3, 'diffuseColor'),
        (3, 'emissiveColor'),
        (3, 'specularColor')]

    def __init__(self, parent):
        super().__init__(parent)
        self.color = model.Material.Color('DefaultVrmlMaterial_{:d}'.format(VrmlMaterial.IDENT))
        self.values = {}
        VrmlMaterial.IDENT += 1

    def read(self, stream, string):
        key = r'([+e\d\-\.]+)'
        for pattern in VrmlMaterial.PATTERNS:
            result = re.search(pattern[1] + r'\s+' + r'\s'.join([key] * pattern[0]), string,
                               re.I | re.S)
            if result is not None:
                values = [float(result.group(x)) for x in range(1, pattern[0] + 1)]
                debug('{:s}Material attribute {:s} found'.format(self.level(), pattern[1]))
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

        debug('{:s}Material properties:'.format(self.level()))
        debug('{:s}Shininess:      {:.3f}'.format(self.level(1), self.color.shininess))
        debug('{:s}Transparency:   {:.3f}'.format(self.level(1), self.color.transparency))
        debug('{:s}Diffuse Color:  {:.3f}, {:.3f}, {:.3f}'.format(
            self.level(1), *self.color.diffuse))
        debug('{:s}Emissive Color: {:.3f}, {:.3f}, {:.3f}'.format(
            self.level(1), *self.color.emissive))
        debug('{:s}Specular Color: {:.3f}, {:.3f}, {:.3f}'.format(
            self.level(1), *self.color.specular))
        debug('{:s}Ambient Color:  {:.3f}, {:.3f}, {:.3f}'.format(
            self.level(1), *self.color.ambient))

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
        self.texture = model.Material.Texture('', 'DefaultVrmlTexture_{:d}'.format(
            VrmlTexture.IDENT))
        VrmlTexture.IDENT += 1

    def read(self, stream, string):
        tmp = re.search(r'url\s+\'([\w\-\.:\/]+)\'', string, re.I | re.S)
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

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# render_ogl41.py
# Copyright (C) 2015 xent
# Project is distributed under the terms of the GNU General Public License v3.0

import math
import os
import sys
import time
import numpy as np

try:
    import geometry
    import model
except ImportError:
    from . import geometry
    from . import model

try:
    # pylint: disable=W0614
    from OpenGL.GL import *
    from OpenGL.GL.shaders import *
    # pylint: enable=W0614
except ImportError:
    print('Error importing OpenGL library')
    sys.exit()

try:
    import glfw
except ImportError:
    print('Error importing GLFW library')
    sys.exit()

try:
    from PIL import Image
    IMAGES_ENABLED = True
except ImportError:
    print('Images disabled, please install imaging library')
    IMAGES_ENABLED = False

DEBUG_ENABLED = False

def debug(text):
    if DEBUG_ENABLED:
        print(text)

def get_opengl_version():
    version = glGetString(GL_VERSION).decode()
    version_str = version.split(' ')[0]
    version_num = [int(part) for part in version_str.split('.')]

    result = version_num[0] * 100
    result += version_num[1] * (10 if version_num[1] < 10 else 1)
    return result

def get_normal(vertices, indices):
    return model.normalize(np.cross(
        vertices[indices[1]] - vertices[indices[0]],
        vertices[indices[2]] - vertices[indices[0]]))

def get_tangent(geo_vertices, tex_vertices, geo_indices, tex_indices):
    return model.normalize(model.tangent(
        geo_vertices[geo_indices[1]] - geo_vertices[geo_indices[0]],
        geo_vertices[geo_indices[2]] - geo_vertices[geo_indices[0]],
        tex_vertices[tex_indices[1]] - tex_vertices[tex_indices[0]],
        tex_vertices[tex_indices[2]] - tex_vertices[tex_indices[0]]))

def generate_mesh_normals(meshes):
    blue_material = model.Material()
    blue_material.color.diffuse = np.array([0.0, 0.0, 1.0])
    normal_length = 0.2

    urchin = model.LineArray(name='Normals')
    urchin.appearance().material = blue_material

    for mesh in meshes:
        if not mesh.appearance().normals:
            continue

        geo_vertices, geo_polygons = mesh.geometry()
        smooth = mesh.appearance().smooth

        if mesh.transform is not None:
            geo_vertices = [mesh.transform.apply(vertex) for vertex in geo_vertices]

        if smooth:
            normals = [np.zeros(3) for i in range(0, len(geo_vertices))]
            for poly in geo_polygons:
                normal = get_normal(geo_vertices, poly)
                for vertex in poly:
                    normals[vertex] += normal
            normals = [model.normalize(vector) for vector in normals]

            for i, vertex in enumerate(geo_vertices):
                last_index = len(urchin.geo_vertices)
                urchin.geo_vertices.append(vertex)
                urchin.geo_vertices.append(vertex + normals[i] * normal_length)
                urchin.geo_polygons.append([last_index, last_index + 1])
        else:
            normals = [get_normal(geo_vertices, poly) for poly in geo_polygons]

            for i, poly in enumerate(geo_polygons):
                position = np.zeros(3)
                for vertex in poly:
                    position += geo_vertices[vertex]
                position /= float(len(poly))
                last_index = len(urchin.geo_vertices)
                urchin.geo_vertices.append(position)
                urchin.geo_vertices.append(position + normals[i] * normal_length)
                urchin.geo_polygons.append([last_index, last_index + 1])

    return urchin if urchin.geo_polygons else None

def build_solid_object_groups(shaders, input_objects):
    # Build meshes for solid objects
    output = []
    objects = [entry for entry in input_objects if entry.style == model.Object.PATCHES]

    # Join meshes with the same material in groups
    mats = [obj.appearance().material for obj in objects]
    keys = []
    for mat in mats:
        if mat not in keys:
            keys.append(mat)
    groups = []
    for key in keys:
        groups.append([obj for obj in objects if obj.appearance().material == key])

    for group in groups:
        appearance = group[0].appearance()
        render_appearance = RenderAppearance.make_from_material(
            shaders, appearance.material, appearance.smooth, appearance.wireframe, appearance.solid)
        output.append(RenderMesh(group, render_appearance))
        debug(f'Render group of {len(group)} mesh(es) created')

    return output

def build_line_object_groups(shaders, input_objects):
    # Build line arrays
    output = []
    objects = [entry for entry in input_objects if entry.style == model.Object.LINES]

    # Build normals for solid objects
    solid_objects = [entry for entry in input_objects if entry.style == model.Object.PATCHES]
    object_normals = generate_mesh_normals(solid_objects)
    if object_normals is not None:
        appearance = RenderAppearance.make_from_material(
            shaders, object_normals.appearance().material)
        output.append(RenderLineArray([object_normals], appearance))

    # Join line arrays in groups
    mats = [obj.appearance().material for obj in objects]
    keys = []
    for mat in mats:
        if mat not in keys:
            keys.append(mat)
    groups = []
    for key in keys:
        groups.append([obj for obj in objects if obj.appearance().material == key])

    for group in groups:
        appearance = group[0].appearance()
        render_appearance = RenderAppearance.make_from_material(shaders, appearance.material)
        output.append(RenderLineArray(group, render_appearance))
        debug(f'Render group of {len(group)} line array(s) created')

    return output

def build_object_groups(shaders, input_objects):
    groups = []
    groups += build_solid_object_groups(shaders, input_objects)
    groups += build_line_object_groups(shaders, input_objects)

    # Sort by material transparency
    groups.sort(key=lambda group: group.appearance.material.color.transparency, reverse=False)
    return groups

class Texture:
    def __init__(self, mode, location, identifier=0, filtering=(GL_LINEAR, GL_LINEAR),
                 repeating=GL_CLAMP_TO_EDGE):
        self.buffer = identifier
        self.mode = mode
        self.location = location
        self.filter_mode = filtering
        self.repeat_mode = repeating

    def activate(self, channel):
        glActiveTexture(GL_TEXTURE0 + channel)
        glBindTexture(self.mode, self.buffer)
        if self.mode == GL_TEXTURE_2D:
            glTexParameteri(self.mode, GL_TEXTURE_WRAP_S, self.repeat_mode)
            glTexParameteri(self.mode, GL_TEXTURE_WRAP_T, self.repeat_mode)
            glTexParameterf(self.mode, GL_TEXTURE_MAG_FILTER, self.filter_mode[0])
            glTexParameterf(self.mode, GL_TEXTURE_MIN_FILTER, self.filter_mode[1])
        glUniform1i(self.location, channel)

    def free(self):
        if self.buffer > 0:
            glDeleteTextures([self.buffer])
            self.buffer = 0
        debug('Overlay freed')


class RenderAppearance:
    class ImageTexture(Texture):
        def __init__(self, location, path):
            super().__init__(GL_TEXTURE_2D, location)

            if IMAGES_ENABLED:
                if not os.path.isfile(path[1]):
                    raise FileNotFoundError()
                image = Image.open(path[1])
                try:
                    self.size, image_data = image.size, image.tobytes('raw', 'RGBA', 0, -1)
                except ValueError:
                    self.size, image_data = image.size, image.tobytes('raw', 'RGBX', 0, -1)
            else:
                self.size = (8, 8)
                black_pixel = bytearray([0x00, 0x00, 0x00, 0xFF])
                purple_pixel = bytearray([0xFF, 0x00, 0xFF, 0xFF])
                width, height = int(self.size[0] / 2), int(self.size[1] / 2)
                image = ((black_pixel + purple_pixel) * width
                         + (purple_pixel + black_pixel) * width) * height
                self.filter_mode = (GL_NEAREST, GL_NEAREST)

            self.buffer = glGenTextures(1)

            glBindTexture(self.mode, self.buffer)
            glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
            glTexImage2D(self.mode, 0, GL_RGBA8, self.size[0], self.size[1], 0, GL_RGBA,
                         GL_UNSIGNED_BYTE, image_data)
            debug(f'Texture loaded, path {path[0]}, width {self.size[0]}, height {self.size[1]}'
                  f', id {self.buffer}')


    def __init__(self, shader=None, material=None, smooth=False, wireframe=False, solid=True):
        if shader is None:
            raise ValueError()

        self.textures = []

        self.shader = shader
        self.smooth = smooth
        self.wireframe = wireframe
        self.solid = False if wireframe else solid
        self.zbuffer = True

        if material is not None:
            self.material = material

            if self.material.diffuse is not None:
                self.textures.append(RenderAppearance.ImageTexture(glGetUniformLocation(
                    self.shader.program, 'diffuseTexture'), self.material.diffuse.path))
            if self.material.normal is not None:
                self.textures.append(RenderAppearance.ImageTexture(glGetUniformLocation(
                    self.shader.program, 'normalTexture'), self.material.normal.path))
            if self.material.specular is not None:
                self.textures.append(RenderAppearance.ImageTexture(glGetUniformLocation(
                    self.shader.program, 'specularTexture'), self.material.specular.path))
        else:
            self.material = model.Material()

    def enable(self, projection_matrix, model_view_matrix, lights):
        self.shader.enable(projection_matrix, model_view_matrix, lights, self.material.color,
                           self.textures)

    @classmethod
    def make_from_material(cls, shaders, material, smooth=False, wireframe=False, solid=True):
        appearance_name = ''
        if material.diffuse is not None:
            appearance_name += 'Diff'
        if material.normal is not None:
            appearance_name += 'Norm'
        if material.specular is not None:
            appearance_name += 'Spec'

        if appearance_name == '':
            appearance_name = 'Colored'

        return cls(shader=shaders[appearance_name], material=material, smooth=smooth,
                   wireframe=wireframe, solid=solid)

    @classmethod
    def make_from_shader(cls, shader, smooth=False, wireframe=False, solid=True):
        return cls(shader=shader, smooth=smooth, wireframe=wireframe, solid=solid)


class RenderObject:
    IDENT = 0

    def __init__(self):
        self.ident = str(RenderObject.IDENT)
        RenderObject.IDENT += 1

    def draw(self, projection_matrix, model_view_matrix, lights, wireframe):
        pass


class RenderLineArray(RenderObject):
    def __init__(self, meshes, appearance=None, transform=None):
        super().__init__()

        self.parts = []
        self.appearance = appearance
        self.transform = transform

        started = time.time()

        primitives = [0]
        for mesh in meshes:
            for poly in mesh.geometry()[1]:
                count = len(poly)
                if count < 2 or count > 2:
                    raise ValueError()
                primitives[count - 2] += 1

        lines = primitives[0] * 2
        length = lines
        self.vertices = np.zeros(length * 3, dtype=np.float32)
        self.colors = np.zeros(length * 3, dtype=np.float32)

        if lines > 0:
            self.parts.append((GL_LINES, 0, lines))

        # Initial positions
        index = [0]

        for mesh in meshes:
            color = mesh.appearance().material.color.diffuse

            geo_vertices, geo_polygons = mesh.geometry()
            if mesh.transform is not None:
                geo_vertices = [mesh.transform.apply(vertex) for vertex in geo_vertices]

            for poly in geo_polygons:
                count = len(poly)
                index_group = count - 2

                offset = index[index_group]
                index[index_group] += count

                for i in range(0, count):
                    start, end = 3 * offset, 3 * (offset + 1)
                    self.vertices[start:end] = geo_vertices[poly[i]][0:3]
                    self.colors[start:end] = color
                    offset += 1

        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)

        self.vertices_vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vertices_vbo)
        glBufferData(GL_ARRAY_BUFFER, self.vertices, GL_STATIC_DRAW)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, None)

        self.colors_vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.colors_vbo)
        glBufferData(GL_ARRAY_BUFFER, self.colors, GL_STATIC_DRAW)
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, None)

        glBindVertexArray(0)

        debug(f'Point cloud created in {time.time() - started}, id {self.ident}'
              f', lines {lines // 2}, vertices {length}')

    def draw(self, projection_matrix, model_view_matrix, lights, wireframe):
        if self.appearance is not None:
            self.appearance.enable(projection_matrix, model_view_matrix, lights)

        glBindVertexArray(self.vao)
        for entry in self.parts:
            glDrawArrays(*entry)
        glBindVertexArray(0)


class RenderMesh(RenderObject):
    def __init__(self, meshes, appearance=None, transform=None):
        super().__init__()

        self.appearance = appearance
        self.transform = transform
        self.parts = []

        textured = meshes[0].is_textured()
        started = time.time()

        primitives = [0, 0]
        for mesh in meshes:
            for poly in mesh.geometry()[1]:
                count = len(poly)
                if count < 3 or count > 4:
                    debug(f'Incorrect polygon size: {count}')
                    continue
                primitives[count - 3] += 1

        triangles, quads = primitives[0] * 3, primitives[1] * 4
        length = triangles + quads
        self.vertices = np.zeros(length * 3, dtype=np.float32)
        self.normals = np.zeros(length * 3, dtype=np.float32)
        self.texels = np.zeros(length * 2, dtype=np.float32) if textured else None
        self.tangents = np.zeros(length * 3, dtype=np.float32) if textured else None

        if triangles > 0:
            self.parts.append((GL_TRIANGLES, 0, triangles))
        if quads > 0:
            self.parts.append((GL_QUADS, triangles, quads))

        smooth = self.appearance.smooth if self.appearance is not None else False

        # Initial positions
        index = [0, triangles]

        for mesh in meshes:
            geo_vertices, geo_polygons = mesh.geometry()
            tex_vertices, tex_polygons = mesh.texture()

            if mesh.transform is not None:
                geo_vertices = [mesh.transform.apply(vertex) for vertex in geo_vertices]

            if smooth:
                normals = [np.zeros(3) for i in range(0, len(geo_vertices))]
                for poly in geo_polygons:
                    normal = get_normal(geo_vertices, poly)
                    for i in poly:
                        normals[i] += normal
                normals = [model.normalize(vector) for vector in normals]

                if textured:
                    tangents = [np.zeros(3) for i in range(0, len(geo_vertices))]
                    for geo_poly, tex_poly in zip(geo_polygons, tex_polygons):
                        tangent = get_tangent(geo_vertices, tex_vertices, geo_poly, tex_poly)
                        for i in geo_poly:
                            tangents[i] += tangent
                    tangents = [model.normalize(vector) for vector in tangents]
            else:
                normals = [get_normal(geo_vertices, poly) for poly in geo_polygons]
                if textured:
                    tangents = []
                    for geo_poly, tex_poly in zip(geo_polygons, tex_polygons):
                        tangents.append(get_tangent(geo_vertices, tex_vertices, geo_poly, tex_poly))

            for i, geo_poly in enumerate(geo_polygons):
                if textured:
                    tex_poly = tex_polygons[i]

                count = len(geo_poly)
                if count < 3 or count > 4:
                    continue
                index_group = count - 3

                offset = index[index_group]
                index[index_group] += count

                for vertex in range(0, count):
                    geo_beg, geo_end = 3 * offset, 3 * (offset + 1)
                    tex_beg, tex_end = 2 * offset, 2 * (offset + 1)

                    self.vertices[geo_beg:geo_end] = geo_vertices[geo_poly[vertex]][0:3]
                    self.normals[geo_beg:geo_end] = \
                        normals[geo_poly[vertex]] if smooth else normals[i]
                    if textured:
                        self.texels[tex_beg:tex_end] = tex_vertices[tex_poly[vertex]]
                        self.tangents[geo_beg:geo_end] = \
                            tangents[geo_poly[vertex]] if smooth else tangents[i]
                    offset += 1

        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)

        self.vertices_vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vertices_vbo)
        glBufferData(GL_ARRAY_BUFFER, self.vertices, GL_STATIC_DRAW)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, None)

        self.normals_vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.normals_vbo)
        glBufferData(GL_ARRAY_BUFFER, self.normals, GL_STATIC_DRAW)
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, None)

        if textured:
            self.texels_vbo = glGenBuffers(1)
            glBindBuffer(GL_ARRAY_BUFFER, self.texels_vbo)
            glBufferData(GL_ARRAY_BUFFER, self.texels, GL_STATIC_DRAW)
            #glBufferData(GL_ARRAY_BUFFER, self.texels, GL_DYNAMIC_DRAW)
            glEnableVertexAttribArray(2)
            glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 0, None)

            self.tangent_vbo = glGenBuffers(1)
            glBindBuffer(GL_ARRAY_BUFFER, self.tangent_vbo)
            glBufferData(GL_ARRAY_BUFFER, self.tangents, GL_STATIC_DRAW)
            glEnableVertexAttribArray(3)
            glVertexAttribPointer(3, 3, GL_FLOAT, GL_FALSE, 0, None)

        glBindVertexArray(0)

        debug(f'Mesh created in {time.time() - started}, id {self.ident}'
              f', triangles {triangles // 3}, quads {quads // 4}, vertices {length}')

    def draw(self, projection_matrix, model_view_matrix, lights, wireframe):
        if self.appearance is not None:
            if self.transform is not None:
                model_view_matrix = np.matmul(self.transform, model_view_matrix)
                lights = [np.matmul(light, np.linalg.inv(self.transform)) for light in lights]

            self.appearance.enable(projection_matrix, model_view_matrix, lights)
            solid = self.appearance.solid
            wireframe = wireframe or self.appearance.wireframe
        else:
            solid = True

        if wireframe:
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
            glDisable(GL_CULL_FACE)
        else:
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
            if solid:
                glEnable(GL_CULL_FACE)
                glCullFace(GL_BACK)
            else:
                glDisable(GL_CULL_FACE)

        glBindVertexArray(self.vao)
        for entry in self.parts:
            glDrawArrays(*entry)
        glBindVertexArray(0)


class ScreenMesh(RenderMesh):
    def __init__(self, meshes):
        super().__init__(meshes)

    def draw(self, projection_matrix=None, model_view_matrix=None, lights=None, wireframe=None):
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        glDisable(GL_CULL_FACE)

        glBindVertexArray(self.vao)
        for entry in self.parts:
            glDrawArrays(*entry)
        glBindVertexArray(0)


class Scene:
    class Camera:
        def __init__(self, fov=90.0, distance=20.0, translation_rate=0.02,
                     rotation_rate=math.pi / 12.0, scaling_rate=1.1):
            self.fov = fov
            self.distance = distance
            self.translation_rate = translation_rate
            self.rotation_rate = rotation_rate
            self.scaling_rate = scaling_rate

            self.pov = np.array([0.0, 0.0, 0.0, 1.0])
            self.camera = np.array([0.0, -self.distance, 0.0, 1.0])
            self.axis = np.array([0.0, 0.0, 1.0, 0.0])

        def reset(self):
            self.pov = np.array([0.0, 0.0, 0.0, 1.0])
            self.camera = np.array([0.0, -self.distance, 0.0, 1.0])
            self.axis = np.array([0.0, 0.0, 1.0, 0.0])

        def front(self):
            distance = np.linalg.norm(self.camera - self.pov)
            self.camera = np.array([0.0, -distance, 0.0, 0.0]) + self.pov # pylint: disable=E1130
            self.axis = np.array([0.0, 0.0, 1.0, 0.0])

        def side(self):
            distance = np.linalg.norm(self.camera - self.pov)
            self.camera = np.array([distance, 0.0, 0.0, 0.0]) + self.pov
            self.axis = np.array([0.0, 0.0, 1.0, 0.0])

        def top(self):
            distance = np.linalg.norm(self.camera - self.pov)
            self.camera = np.array([0.0, 0.0, distance, 0.0]) + self.pov
            self.axis = np.array([0.0, 1.0, 0.0, 0.0])

        def rotate(self, hrot, vrot):
            camera = self.camera - self.pov
            axis = self.axis
            if hrot != 0.0:
                horiz_rotation_matrix = np.array([
                    [ math.cos(hrot), math.sin(hrot), 0.0, 0.0],
                    [-math.sin(hrot), math.cos(hrot), 0.0, 0.0],
                    [            0.0,            0.0, 1.0, 0.0],
                    [            0.0,            0.0, 0.0, 1.0]])
                camera = np.matmul(camera, horiz_rotation_matrix)
                axis = np.matmul(axis, horiz_rotation_matrix)
            if vrot != 0.0:
                normal = np.cross(camera[0:3], self.axis[0:3])
                normal /= np.linalg.norm(normal)
                vert_rotation_matrix = model.make_rotation_matrix(normal, vrot).transpose()
                camera = np.matmul(camera, vert_rotation_matrix)
                axis = np.matmul(axis, vert_rotation_matrix)
            self.camera = camera + self.pov
            self.axis = axis

        def move(self, x, y): # pylint: disable=C0103
            camera = self.camera - self.pov
            normal = np.cross(self.axis[0:3], camera[0:3])
            normal /= np.linalg.norm(normal)
            distance = np.linalg.norm(camera)
            width = 2.0 * math.tan(self.fov / 2.0) * distance

            offset = normal * (-x / width) + self.axis[0:3] * (-y / width)
            offset *= distance * self.translation_rate
            offset = np.array([*offset, 0.0])

            self.camera += offset
            self.pov += offset

        def zoom(self, z): # pylint: disable=C0103
            scale = np.array([z, z, z, 1.0])
            self.camera = (self.camera - self.pov) * scale + self.pov

        def zoom_in(self):
            self.zoom(1.0 / self.scaling_rate)

        def zoom_out(self):
            self.zoom(self.scaling_rate)


    def __init__(self):
        self.ortho = False
        self.depth = (0.001, 1000.0)
        self.camera = Scene.Camera()
        self.viewport = (640, 480) # FIXME
        self.update_matrix(self.viewport)

        self.lights = [
            [+50.0, +50.0, +50.0, 1.0],
            [-50.0, -50.0, -50.0, 1.0]]

    def update_matrix(self, viewport):
        aspect = float(viewport[0]) / float(viewport[1])
        if self.ortho:
            distance = np.linalg.norm(self.camera.camera - self.camera.pov)
            width = 1.0 / math.tan(self.camera.fov / 2.0) * distance
            area = (width, width / aspect)
            self.projection_matrix = model.create_orthographic_matrix(area, self.depth)
        else:
            self.projection_matrix = model.create_perspective_matrix(
                aspect, self.camera.fov, self.depth)

        self.model_view_matrix = model.create_model_view_matrix(
            self.camera.camera[0:3], self.camera.pov[0:3], self.camera.axis[0:3])

    @staticmethod
    def reset_shaders():
        glUseProgram(0)


class Shader:
    IDENT = 0

    def __init__(self):
        self.ident = Shader.IDENT
        Shader.IDENT += 1

        self.dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'shaders_ogl41')
        self.program = None

    def create(self, vert_shader, frag_shader, geometry_shader=None, control_shader=None,
               eval_shader=None):
        try:
            shaders = []
            shaders.append(compileShader(vert_shader, GL_VERTEX_SHADER))
            shaders.append(compileShader(frag_shader, GL_FRAGMENT_SHADER))
            if geometry_shader is not None:
                shaders.append(compileShader(geometry_shader, GL_GEOMETRY_SHADER))
            if control_shader is not None:
                shaders.append(compileShader(control_shader, GL_TESS_CONTROL_SHADER))
            if eval_shader is not None:
                shaders.append(compileShader(eval_shader, GL_TESS_EVALUATION_SHADER))
            self.program = compileProgram(*shaders)
            debug(f'Shader {self.ident} compiled')
        except RuntimeError as run_error:
            print(run_error.args[0]) # Print error log
            print(f'Shader {self.ident} compilation failed in {str(run_error.args[2])}')
            sys.exit()

    def enable_program(self):
        glUseProgram(self.program)


class BaseModelShader(Shader):
    def __init__(self, texture, normal, specular):
        super().__init__()

        if self.program is None:
            flags = ['#define LIGHT_COUNT 2']
            if texture:
                flags += ['#define DIFFUSE_MAP']
            if normal:
                flags += ['#define NORMAL_MAP']
            if specular:
                flags += ['#define SPECULAR_MAP']

            def load_file(path):
                with open(os.path.join(self.dir, path), 'rb') as file:
                    source = file.read().decode('utf-8').split('\n')
                    source = [source[0]] + flags + source[1:]
                    return '\n'.join(source)

            self.create(load_file('default.vert'), load_file('default.frag'))

        self.light_ambient = 0.1
        self.light_diffuse = np.array([[0.9, 0.9, 0.9], [0.9, 0.9, 0.9]], np.float32)

        self.projection_loc = glGetUniformLocation(self.program, 'projectionMatrix')
        self.model_view_loc = glGetUniformLocation(self.program, 'modelViewMatrix')
        self.normal_loc = glGetUniformLocation(self.program, 'normalMatrix')

        self.light_loc = glGetUniformLocation(self.program, 'lightPosition')
        self.light_diffuse_loc = glGetUniformLocation(self.program, 'lightDiffuseColor')
        self.light_ambient_loc = glGetUniformLocation(self.program, 'lightAmbientIntensity')

    def enable(self, projection_matrix, model_view_matrix, lights, _1, _2):
        self.enable_program()

        # Set precalculated matrices
        normal_matrix = np.transpose(np.linalg.inv(model_view_matrix))
        glUniformMatrix4fv(self.projection_loc, 1, GL_FALSE,
                           np.array(projection_matrix, np.float32))
        glUniformMatrix4fv(self.model_view_loc, 1, GL_FALSE,
                           np.array(model_view_matrix, np.float32))
        glUniformMatrix4fv(self.normal_loc, 1, GL_FALSE,
                           np.array(normal_matrix, np.float32))

        # Set precalculated light positions, configure colors of diffuse and ambient lighting
        lights = np.array([np.matmul(light, model_view_matrix)[0:3] for light in lights],
                           np.float32)
        glUniform3fv(self.light_loc, len(lights), lights)
        glUniform3fv(self.light_diffuse_loc, 2, self.light_diffuse)
        glUniform1f(self.light_ambient_loc, self.light_ambient)


class ModelShader(BaseModelShader):
    def __init__(self, texture=False, normal=False, specular=False):
        super().__init__(texture, normal, specular)

        self.diffuse_color_loc = glGetUniformLocation(self.program, 'materialDiffuseColor')
        self.specular_color_loc = glGetUniformLocation(self.program, 'materialSpecularColor')
        self.emissive_color_loc = glGetUniformLocation(self.program, 'materialEmissiveColor')
        self.shininess_loc = glGetUniformLocation(self.program, 'materialShininess')

    def enable(self, projection_matrix, model_view_matrix, lights, colors, textures):
        super().enable(projection_matrix, model_view_matrix, lights, colors, textures)

        glUniform4fv(self.diffuse_color_loc, 1, list(colors.diffuse) + [1.0 - colors.transparency])
        glUniform3fv(self.specular_color_loc, 1, colors.specular)
        glUniform3fv(self.emissive_color_loc, 1, colors.emissive)
        glUniform1f(self.shininess_loc, colors.shininess * 128.0)

        for i, texture in enumerate(textures):
            texture.activate(i)


class UnlitModelShader(Shader):
    def __init__(self):
        super().__init__()

        if self.program is None:
            def load_file(path):
                with open(os.path.join(self.dir, path), 'rb') as file:
                    return file.read().decode('utf-8')

            self.create(load_file('unlit.vert'), load_file('unlit.frag'))

        self.projection_loc = glGetUniformLocation(self.program, 'projectionMatrix')
        self.model_view_loc = glGetUniformLocation(self.program, 'modelViewMatrix')
        self.normal_loc = glGetUniformLocation(self.program, 'normalMatrix')

    def enable(self, projection_matrix, model_view_matrix, _1, _2, _3):
        self.enable_program()

        normal_matrix = np.transpose(np.linalg.inv(model_view_matrix))
        glUniformMatrix4fv(self.projection_loc, 1, GL_FALSE,
                           np.array(projection_matrix, np.float32))
        glUniformMatrix4fv(self.model_view_loc, 1, GL_FALSE,
                           np.array(model_view_matrix, np.float32))
        glUniformMatrix4fv(self.normal_loc, 1, GL_FALSE,
                           np.array(normal_matrix, np.float32))


class SystemShader(Shader):
    def __init__(self, prefix='', flags=None):
        super().__init__()

        if flags is None:
            flags = []

        if self.program is None and prefix != '':
            def load_file(path, flags):
                with open(os.path.join(self.dir, path), 'rb') as file:
                    source = file.read().decode('utf-8').split('\n')
                    source = [source[0]] + flags + source[1:]
                    return '\n'.join(source)

            self.create(load_file(prefix + '.vert', flags), load_file(prefix + '.frag', flags))

        self.projection_loc = glGetUniformLocation(self.program, 'projectionMatrix')
        self.model_view_loc = glGetUniformLocation(self.program, 'modelViewMatrix')

        self.projection_matrix = model.create_orthographic_matrix(
            area=(1.0, 1.0),
            distance=(0.001, 1000.0))
        self.model_view_matrix = model.create_model_view_matrix(
            eye=np.array([0.0, 0.0, 1.0]),
            center=np.zeros(3),
            z_axis=np.array([0.0, 1.0, 0.0]))

    def setup_projections(self):
        self.enable_program()

        glUniformMatrix4fv(self.projection_loc, 1, GL_FALSE,
                           np.array(self.projection_matrix, np.float32))
        glUniformMatrix4fv(self.model_view_loc, 1, GL_FALSE,
                           np.array(self.model_view_matrix, np.float32))


class BackgroundShader(SystemShader):
    def __init__(self):
        super().__init__('background')

    def enable(self):
        self.setup_projections()


class MergeShader(SystemShader):
    def __init__(self, antialiasing):
        flags = ['#define AA_SAMPLES ' + str(antialiasing)] if antialiasing > 0 else []
        super().__init__('merge', flags)

        mode = GL_TEXTURE_2D_MULTISAMPLE if antialiasing > 0 else GL_TEXTURE_2D
        self.color_texture = Texture(mode, glGetUniformLocation(self.program, 'colorTexture'))

    def enable(self, color_buffer):
        self.setup_projections()

        self.color_texture.buffer = color_buffer
        self.color_texture.activate(0)
        if self.color_texture.mode == GL_TEXTURE_2D:
            glGenerateMipmap(GL_TEXTURE_2D)


class BlurShader(SystemShader):
    def __init__(self, masked):
        flags = ['#define MASKED'] if masked else []
        super().__init__('blur', flags)

        self.color_texture = Texture(mode=GL_TEXTURE_2D, location=glGetUniformLocation(
            self.program, 'colorTexture'))
        self.direction_loc = glGetUniformLocation(self.program, 'direction')

        if masked:
            self.mask_texture = Texture(mode=GL_TEXTURE_2D, location=glGetUniformLocation(
                self.program, 'maskTexture'), filtering=(GL_NEAREST, GL_NEAREST))
            self.source_texture = Texture(mode=GL_TEXTURE_2D, location=glGetUniformLocation(
                self.program, 'sourceTexture'))
        else:
            self.mask_texture = None
            self.source_texture = None

    def enable(self, resolution, direction, color_buffer, source_buffer=None, mask_buffer=None):
        self.setup_projections()

        glUniform2fv(self.direction_loc, 1, direction / resolution)

        self.color_texture.buffer = color_buffer
        self.color_texture.activate(0)

        if self.mask_texture is not None:
            self.mask_texture.buffer = mask_buffer
            self.mask_texture.activate(1)
        if self.source_texture is not None:
            self.source_texture.buffer = source_buffer
            self.source_texture.activate(2)


class Framebuffer:
    def __init__(self, size, antialiasing, depth):
        # Color buffer
        self.color = glGenTextures(1)

        if antialiasing > 0:
            glBindTexture(GL_TEXTURE_2D_MULTISAMPLE, self.color)
            glTexImage2DMultisample(GL_TEXTURE_2D_MULTISAMPLE, antialiasing, GL_RGBA8,
                                    size[0], size[1], GL_TRUE)
            glBindTexture(GL_TEXTURE_2D_MULTISAMPLE, 0)
        else:
            glBindTexture(GL_TEXTURE_2D, self.color)
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, size[0], size[1], 0, GL_RGBA, GL_UNSIGNED_BYTE,
                         None)
            glBindTexture(GL_TEXTURE_2D, 0)

        if depth:
            # Depth buffer
            self.depth = glGenRenderbuffers(1)

            if antialiasing > 0:
                glBindRenderbuffer(GL_RENDERBUFFER, self.depth)
                glRenderbufferStorageMultisample(GL_RENDERBUFFER, antialiasing, GL_DEPTH_COMPONENT,
                                                 size[0], size[1])
                glBindRenderbuffer(GL_RENDERBUFFER, 0)
            else:
                glBindRenderbuffer(GL_RENDERBUFFER, self.depth)
                glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, size[0], size[1])
                glBindRenderbuffer(GL_RENDERBUFFER, 0)
        else:
            self.depth = 0

        # Create framebuffer
        self.buffer = glGenFramebuffers(1)
        glBindFramebuffer(GL_FRAMEBUFFER, self.buffer)

        if antialiasing > 0:
            glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D_MULTISAMPLE,
                                   self.color, 0)
        else:
            glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D,
                                   self.color, 0)
        if depth:
            glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER,
                                      self.depth)

        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        debug(f'Framebuffer built, size {size[0]}x{size[1]}'
              f', color {self.color}, depth {self.depth}')

    def free(self):
        if self.color:
            glDeleteTextures([self.color])
            self.color = 0
        if self.depth > 0:
            glDeleteRenderbuffers(1, [self.depth])
            self.depth = 0
        if self.buffer > 0:
            glDeleteFramebuffers(1, [self.buffer])
            self.buffer = 0

        debug('Framebuffer freed')


class Render(Scene):
    class ShaderStorage:
        def __init__(self, antialiasing=0, overlay=False):
            self.shaders = {}
            self.background = None
            self.blur = None
            self.blur_masked = None
            self.merge = None

            self.background = BackgroundShader()

            self.shaders['Colored'] = ModelShader()
            self.shaders['Unlit'] = UnlitModelShader()

            self.shaders['Diff'] = ModelShader(texture=True, normal=False, specular=False)
            self.shaders['Norm'] = ModelShader(texture=False, normal=True, specular=False)
            self.shaders['Spec'] = ModelShader(texture=False, normal=False, specular=True)
            self.shaders['DiffNorm'] = ModelShader(texture=True, normal=True, specular=False)
            self.shaders['DiffSpec'] = ModelShader(texture=True, normal=False, specular=True)
            self.shaders['NormSpec'] = ModelShader(texture=False, normal=True, specular=True)
            self.shaders['DiffNormSpec'] = ModelShader(texture=True, normal=True, specular=True)

            legacy = get_opengl_version() < 310

            if not legacy and (overlay or antialiasing > 0):
                self.merge = MergeShader(antialiasing=antialiasing)

            if not legacy and overlay:
                self.blur = BlurShader(masked=False)
                self.blur_masked = BlurShader(masked=True)


    def __init__(self, objects=None, options=None):
        super().__init__()

        if options is not None:
            self.parse_options(options)
        else:
            self.antialiasing = 0
            self.overlay = False
            self.wireframe = False
            self.use_framebuffers = False

        self.camera_cursor = [0.0, 0.0]
        self.camera_move = False
        self.camera_rotate = False
        self.redisplay = True
        self.title_text = 'OpenGL 4.1 render'

        self.framebuffers = []
        self.overlay_mask = None

        glfw.init()

        try:
            self.window = glfw.create_window(*self.viewport, self.title_text, None, None)
            glfw.make_context_current(self.window)
        except glfw.GLFWError:
            print('Window initialization failed')
            glfw.terminate()
            sys.exit()

        self.init_graphics()
        self.update_matrix(self.viewport)

        glfw.set_key_callback(self.window, self.handle_key_event)
        glfw.set_mouse_button_callback(self.window, self.handle_mouse_button_event)
        glfw.set_cursor_pos_callback(self.window, self.handle_cursor_move_event)
        glfw.set_scroll_callback(self.window, self.handle_scroll_event)
        glfw.set_window_refresh_callback(self.window, self.handle_resize_event)

        self.objects = set()
        if objects is not None:
            self.append_render_objects(self.make_render_objects(objects))

    def redraw(self):
        self.redisplay = True
        glfw.post_empty_event()

    def run(self):
        while not glfw.window_should_close(self.window):
            if self.redisplay:
                self.redisplay = False
                self.draw_scene()
            glfw.wait_events()
        glfw.terminate()

    def parse_options(self, options):
        self.antialiasing = 0 if 'antialiasing' not in options else options['antialiasing']
        self.overlay = False if 'overlay' not in options else options['overlay']
        self.wireframe = False if 'wireframe' not in options else options['wireframe']
        if 'size' in options:
            self.viewport = options['size']
        self.use_framebuffers = self.antialiasing > 0 or self.overlay

    def make_render_objects(self, objects):
        return build_object_groups(self.shader_storage.shaders, objects)

    def append_render_objects(self, render_objects):
        for entry in render_objects:
            self.objects.add(entry)

    def remove_render_objects(self, render_objects):
        for entry in render_objects:
            self.objects.discard(entry)

    def init_graphics(self):
        glClearColor(0.0, 0.0, 0.0, 1.0)
        glClearDepth(1.0)
        glDepthFunc(GL_LESS)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_TEXTURE_2D)

        self.shader_storage = Render.ShaderStorage(self.antialiasing, self.overlay)
        self.screen_plane = ScreenMesh([geometry.Plane((2.0, 2.0), (1, 1))])

        legacy = get_opengl_version() < 310

        if not legacy and self.use_framebuffers:
            self.init_framebuffers()
        else:
            self.use_framebuffers = False

        if not legacy and self.overlay:
            self.init_overlay()
        else:
            self.overlay = False

    def init_framebuffers(self):
        for framebuffer in self.framebuffers:
            framebuffer.free()
        self.framebuffers = []
        self.framebuffers.append(Framebuffer(self.viewport, self.antialiasing, True))
        self.framebuffers.append(Framebuffer(self.viewport, 0, False))
        self.framebuffers.append(Framebuffer(self.viewport, 0, False))

    def init_overlay(self):
        # Buffer format is R5G5B5A1
        masked, unmasked = bytearray([0x21, 0x84]), bytearray([0xFE, 0xFF])
        part = 0.25

        if self.overlay_mask is not None:
            self.overlay_mask.free()
            self.overlay_mask = None

        mask_buffer = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, mask_buffer)
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1)

        x, y = self.viewport # pylint: disable=C0103
        mask_data = masked * (x * int(y * part)) + unmasked * (x * int(y * (1.0 - part)))

        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB5_A1, x, y, 0, GL_RGBA, GL_UNSIGNED_SHORT_5_5_5_1,
                     mask_data)
        glBindTexture(GL_TEXTURE_2D, 0)
        self.overlay_mask = Texture(mode=GL_TEXTURE_2D, location=0, identifier=mask_buffer)
        debug(f'Overlay built, size {x}x{y}, texture {mask_buffer}')

    def draw_scene(self):
        # First pass
        if self.use_framebuffers:
            glBindFramebuffer(GL_FRAMEBUFFER, self.framebuffers[0].buffer)
            glDrawBuffers(1, [GL_COLOR_ATTACHMENT0])

            if self.antialiasing > 0:
                glEnable(GL_MULTISAMPLE)
        else:
            glBindFramebuffer(GL_FRAMEBUFFER, 0)

        # Enable writing to depth mask and clear it
        glDepthMask(GL_TRUE)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        Render.reset_shaders()

        # Draw background, do not use depth mask
        glDepthMask(GL_FALSE)
        glDisable(GL_DEPTH_TEST)
        self.shader_storage.background.enable()
        self.screen_plane.draw()

        # Draw other objects, use depth mask and enable depth test
        glDepthMask(GL_TRUE)
        glEnable(GL_DEPTH_TEST)
        for entry in self.objects:
            entry.draw(self.projection_matrix, self.model_view_matrix, self.lights, self.wireframe)

        # Second pass, do not use depth mask, depth test is disabled
        glDepthMask(GL_FALSE)
        glDisable(GL_DEPTH_TEST)
        if self.use_framebuffers:
            if self.antialiasing > 0:
                glDisable(GL_MULTISAMPLE)

            if self.overlay:
                glBindFramebuffer(GL_FRAMEBUFFER, self.framebuffers[1].buffer)
                self.shader_storage.merge.enable(self.framebuffers[0].color)
                self.screen_plane.draw()

                glBindFramebuffer(GL_FRAMEBUFFER, self.framebuffers[2].buffer)
                self.shader_storage.blur.enable(self.viewport, np.array([0.0, 1.0]),
                                                self.framebuffers[1].color)
                self.screen_plane.draw()

                glBindFramebuffer(GL_FRAMEBUFFER, 0)
                self.shader_storage.blur_masked.enable(self.viewport, np.array([1.0, 0.0]),
                                                       self.framebuffers[2].color,
                                                       self.framebuffers[1].color,
                                                       self.overlay_mask.buffer)
                self.screen_plane.draw()
            else:
                glBindFramebuffer(GL_FRAMEBUFFER, 0)
                self.shader_storage.merge.enable(self.framebuffers[0].color)
                self.screen_plane.draw()

        glfw.swap_buffers(self.window)

    def handle_cursor_move_event(self, _, xpos, ypos):
        updated = False

        if self.camera_rotate:
            hrot = (self.camera_cursor[0] - xpos) / 100.0
            vrot = (ypos - self.camera_cursor[1]) / 100.0
            self.camera.rotate(hrot, vrot)
            self.camera_cursor = [xpos, ypos]
            updated = True

        if self.camera_move:
            self.camera.move(xpos - self.camera_cursor[0], self.camera_cursor[1] - ypos)
            self.camera_cursor = [xpos, ypos]
            updated = True

        if updated:
            self.update_matrix(self.viewport)
            self.redisplay = True

    def handle_key_event(self, window, key, _1, action, _2):
        redisplay = True
        updated = True

        if key in (glfw.KEY_Q, glfw.KEY_ESCAPE) and action == glfw.PRESS:
            glfw.set_window_should_close(window, GL_TRUE)
            updated = False
        elif key == glfw.KEY_KP_1 and action == glfw.PRESS:
            self.camera.front()
        elif key == glfw.KEY_KP_2 and action != glfw.RELEASE:
            self.camera.rotate(0.0, -self.camera.rotation_rate)
        elif key == glfw.KEY_KP_3 and action == glfw.PRESS:
            self.camera.side()
        elif key == glfw.KEY_KP_4 and action != glfw.RELEASE:
            self.camera.rotate(-self.camera.rotation_rate, 0.0)
        elif key == glfw.KEY_KP_5 and action == glfw.PRESS:
            self.ortho = not self.ortho
        elif key == glfw.KEY_KP_6 and action != glfw.RELEASE:
            self.camera.rotate(self.camera.rotation_rate, 0.0)
        elif key == glfw.KEY_KP_7 and action == glfw.PRESS:
            self.camera.top()
        elif key == glfw.KEY_KP_8 and action != glfw.RELEASE:
            self.camera.rotate(0.0, self.camera.rotation_rate)
        elif key == glfw.KEY_KP_9 and action == glfw.PRESS:
            self.camera.rotate(math.pi, 0.0)
        elif key == glfw.KEY_KP_DECIMAL and action == glfw.PRESS:
            self.camera.reset()
        elif key == glfw.KEY_KP_SUBTRACT and action == glfw.PRESS:
            self.camera.zoom_out()
        elif key == glfw.KEY_KP_ADD and action == glfw.PRESS:
            self.camera.zoom_in()
        elif key == glfw.KEY_Z and action == glfw.PRESS:
            self.wireframe = not self.wireframe
            updated = False
        else:
            redisplay = False
            updated = False

        if updated:
            self.update_matrix(self.viewport)
        if redisplay:
            self.redisplay = True

    def handle_mouse_button_event(self, window, button, action, _):
        updated = False

        if button == glfw.MOUSE_BUTTON_LEFT:
            self.camera_rotate = action == glfw.PRESS
            if self.camera_rotate:
                pos = glfw.get_cursor_pos(window)
                self.camera_cursor = [*pos]
                updated = True
        elif button == glfw.MOUSE_BUTTON_MIDDLE:
            self.camera_move = action == glfw.PRESS
            if self.camera_move:
                pos = glfw.get_cursor_pos(window)
                self.camera_cursor = [*pos]
                updated = True
        if updated:
            self.update_matrix(self.viewport)
            self.redisplay = True

    def handle_resize_event(self, window):
        w, h = glfw.get_framebuffer_size(window) # pylint: disable=invalid-name
        viewport = (int(max(1, w)), int(max(1, h)))

        if viewport != self.viewport:
            self.viewport = viewport
            self.update_matrix(self.viewport)

            if self.use_framebuffers:
                self.init_framebuffers()
            if self.overlay:
                self.init_overlay()

            glViewport(0, 0, *self.viewport)
            self.redisplay = True

    def handle_scroll_event(self, _1, _2, yoffset):
        updated = False

        if yoffset > 0.0:
            self.camera.zoom_in()
            updated = True
        elif yoffset < 0.0:
            self.camera.zoom_out()
            updated = True

        if updated:
            self.update_matrix(self.viewport)
            self.redisplay = True

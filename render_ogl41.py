#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# render_ogl41.py
# Copyright (C) 2015 xent
# Project is distributed under the terms of the GNU General Public License v3.0

import math
import numpy
import os
import time

try:
    import geometry
    import model
except ImportError:
    from . import geometry
    from . import model

try:
    from OpenGL.GL import *
    from OpenGL.GLU import *
    from OpenGL.GLUT import *
    from OpenGL.GL.shaders import *
except:
    print('Error importing OpenGL libraries')
    exit()

try:
    from PIL import Image
    imagesEnabled = True
except:
    imagesEnabled = False
    print('Images disabled, please install imaging library')

debugEnabled = False

def debug(text):
    if debugEnabled:
        print(text)

def generateMeshNormals(meshes):
    blueMaterial = model.Material()
    blueMaterial.color.diffuse = numpy.array([0.0, 0.0, 1.0])
    normalLength = 0.2

    urchin = model.LineArray(name='Normals')
    urchin.appearance().material = blueMaterial

    for mesh in meshes:
        if not mesh.appearance().normals:
            continue

        geoVertices, geoPolygons = mesh.geometry()
        smooth = mesh.appearance().smooth

        vertices = geoVertices if mesh.transform is None else [mesh.transform.apply(v) for v in geoVertices]

        def getNormal(points):
            return model.normalize(numpy.cross(
                    vertices[points[1]] - vertices[points[0]],
                    vertices[points[2]] - vertices[points[0]]))

        if smooth:
            normals = [numpy.zeros(3) for i in range(0, len(vertices))]
            for poly in geoPolygons:
                normal = getNormal(poly)
                for vertex in poly:
                    normals[vertex] += normal
            normals = [model.normalize(vector) for vector in normals]

            for i in range(0, len(vertices)):
                lastIndex = len(urchin.geoVertices)
                urchin.geoVertices.append(vertices[i])
                urchin.geoVertices.append(vertices[i] + normals[i] * normalLength)
                urchin.geoPolygons.append([lastIndex, lastIndex + 1])
        else:
            normals = [getNormal(gp) for gp in geoPolygons]

            for i in range(0, len(geoPolygons)):
                gp = geoPolygons[i]
                position = numpy.zeros(3)
                for vertex in gp:
                    position += vertices[vertex]
                position /= float(len(gp))
                lastIndex = len(urchin.geoVertices)
                urchin.geoVertices.append(position)
                urchin.geoVertices.append(position + normals[i] * normalLength)
                urchin.geoPolygons.append([lastIndex, lastIndex + 1])

    return [] if len(urchin.geoPolygons) == 0 else [urchin]

def buildObjectGroups(shaders, inputObjects):
    renderObjects = []
    objects = inputObjects

    # Render meshes
    meshes = [entry for entry in objects if entry.style == model.Object.PATCHES]
    objects.extend(generateMeshNormals(meshes))

    # Join meshes with the same material in groups
    meshMats = [mesh.appearance().material for mesh in meshes]
    meshKeys = []
    [meshKeys.append(mat) for mat in meshMats if mat not in meshKeys]
    meshGroups = [[mesh for mesh in meshes if mesh.appearance().material == key] for key in meshKeys]

    for group in meshGroups:
        appearance = group[0].appearance()
        renderAppearance = RenderAppearance.makeFromMaterial(shaders, appearance.material,
                appearance.smooth, appearance.wireframe, appearance.solid)
        renderObjects.append(RenderMesh(group, renderAppearance))
        debug('Render group of {:d} mesh(es) created'.format(len(group)))

    # Render line arrays
    arrays = [entry for entry in objects if entry.style == model.Object.LINES]

    # Join line arrays in groups
    arrayMats = [array.appearance().material for array in arrays]
    arrayKeys = []
    [arrayKeys.append(mat) for mat in arrayMats if mat not in arrayKeys]
    arrayGroups = [[array for array in arrays if array.appearance().material == key] for key in arrayKeys]

    for group in arrayGroups:
        appearance = group[0].appearance()
        renderAppearance = RenderAppearance.makeFromMaterial(shaders, appearance.material)
        renderObjects.append(RenderLineArray(group, renderAppearance))
        debug('Render group of {:d} line array(s) created'.format(len(group)))

    # Sort by material transparency
    sortedObjects = [entry for entry in renderObjects if entry.appearance.material.color.transparency <= 0.001]
    sortedObjects += [entry for entry in renderObjects if entry.appearance.material.color.transparency > 0.001]

    return sortedObjects


class Texture:
    def __init__(self, mode, location, identifier=0, filtering=(GL_LINEAR, GL_LINEAR), repeating=GL_CLAMP_TO_EDGE):
        self.buffer = identifier
        self.mode = mode
        self.location = location
        self.filterMode = filtering
        self.repeatMode = repeating

    def free(self):
        if self.buffer > 0:
            glDeleteTextures([self.buffer])
            self.buffer = 0
        debug('Overlay freed')


class RenderAppearance:
    class ImageTexture(Texture):
        def __init__(self, location, path):
            super().__init__(GL_TEXTURE_2D, location)

            if imagesEnabled:
                if not os.path.isfile(path[1]):
                    raise Exception()
                im = Image.open(path[1])
                try:
                    self.size, image = im.size, im.tobytes('raw', 'RGBA', 0, -1)
                except ValueError:
                    self.size, image = im.size, im.tobytes('raw', 'RGBX', 0, -1)
            else:
                self.size = (8, 8)
                pBlack, pPurple = bytearray([0x00, 0x00, 0x00, 0xFF]), bytearray([0xFF, 0x00, 0xFF, 0xFF])
                width, height = self.size[0] / 2, self.size[1] / 2
                image = ((pBlack + pPurple) * width + (pPurple + pBlack) * width) * height
                self.filterMode = (GL_NEAREST, GL_NEAREST)

            self.buffer = glGenTextures(1)

            glBindTexture(self.mode, self.buffer)
            glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
            glTexImage2D(self.mode, 0, GL_RGBA8, self.size[0], self.size[1], 0, GL_RGBA, GL_UNSIGNED_BYTE, image)
            debug('Texture loaded: {:s}, width: {:d}, height: {:d}, id: {:d}'.format(
                    path[0], self.size[0], self.size[1], self.buffer))


    def __init__(self, shader=None, material=None, smooth=False, wireframe=False, solid=True):
        if shader is None:
            raise Exception()

        self.textures = []

        self.shader = shader
        self.smooth = smooth
        self.wireframe = wireframe
        self.solid = False if wireframe else solid
        self.zbuffer = True

        if material is not None:
            self.material = material

            if self.material.diffuse is not None:
                self.textures.append(RenderAppearance.ImageTexture(glGetUniformLocation(self.shader.program,
                        'diffuseTexture'), self.material.diffuse.path))
            if self.material.normal is not None:
                self.textures.append(RenderAppearance.ImageTexture(glGetUniformLocation(self.shader.program,
                        'normalTexture'), self.material.normal.path))
            if self.material.specular is not None:
                self.textures.append(RenderAppearance.ImageTexture(glGetUniformLocation(self.shader.program,
                        'specularTexture'), self.material.specular.path))
        else:
            self.material = model.Material()

    def enable(self, projectionMatrix, modelViewMatrix, lights):
        self.shader.enable(projectionMatrix, modelViewMatrix, lights, self.material.color, self.textures)

    @classmethod
    def makeFromMaterial(cls, shaders, material, smooth=False, wireframe=False, solid=True):
        name = ''
        if material.diffuse is not None:
            name += 'Diff'
        if material.normal is not None:
            name += 'Norm'
        if material.specular is not None:
            name += 'Spec'

        if name == '':
            name = 'Colored'

        return cls(shader=shaders[name], material=material, smooth=smooth, wireframe=wireframe, solid=solid)

    @classmethod
    def makeFromShader(cls, shader, smooth=False, wireframe=False, solid=True):
        return cls(shader=shader, smooth=smooth, wireframe=wireframe, solid=solid)


class RenderObject:
    class Faceset:
        def __init__(self, mode, index, count):
            self.mode = mode
            self.index = index
            self.count = count


    IDENT = 0

    def __init__(self):
        self.ident = str(RenderObject.IDENT)
        RenderObject.IDENT += 1

    def draw(self, projectionMatrix, modelViewMatrix, lights, wireframe):
        pass


class RenderLineArray(RenderObject):
    def __init__(self, meshes, appearance):
        super().__init__()

        self.parts = []
        self.appearance = appearance

        started = time.time()

        primitives = [0]
        for mesh in meshes:
            for poly in mesh.geometry()[1]:
                count = len(poly)
                if count < 2 or count > 2:
                    raise Exception()
                primitives[count - 2] += 1

        lines = primitives[0] * 2
        length = lines
        self.vertices = numpy.zeros(length * 3, dtype=numpy.float32)
        self.colors = numpy.zeros(length * 3, dtype=numpy.float32)

        if lines > 0:
            self.parts.append(RenderMesh.Faceset(GL_LINES, 0, lines))

        # Initial positions
        index = [0]

        for mesh in meshes:
            geoVertices, geoPolygons = mesh.geometry()
            color = mesh.appearance().material.color.diffuse

            vertices = geoVertices if mesh.transform is None else [mesh.transform.apply(v) for v in geoVertices]

            for gp in geoPolygons:
                count = len(gp)
                indexGroup = count - 2

                offset = index[indexGroup]
                index[indexGroup] += count

                for vertex in range(0, count):
                    start, end = 3 * offset, 3 * (offset + 1)
                    self.vertices[start:end] = vertices[gp[vertex]][0:3]
                    self.colors[start:end] = color
                    offset += 1

        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)

        self.verticesVBO = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.verticesVBO)
        glBufferData(GL_ARRAY_BUFFER, self.vertices, GL_STATIC_DRAW)
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, None)

        self.colorsVBO = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.colorsVBO)
        glBufferData(GL_ARRAY_BUFFER, self.colors, GL_STATIC_DRAW)
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, None)

        glBindVertexArray(0)

        debug('Point cloud created in {:f}, id {:s}, lines {:d}, vertices {:d}'.format(
                time.time() - started, self.ident, int(lines / 2), length))

    def draw(self, projectionMatrix, modelViewMatrix, lights, wireframe):
        self.appearance.enable(projectionMatrix, modelViewMatrix, lights)

        glBindVertexArray(self.vao)
        for entry in self.parts:
            glDrawArrays(entry.mode, entry.index, entry.count)
        glBindVertexArray(0)


class RenderMesh(RenderObject):
    def __init__(self, meshes, appearance=None, transform=None):
        super().__init__()

        self.appearance = appearance
        self.transform = transform
        self.parts = []

        textured = meshes[0].isTextured()
        started = time.time()

        primitives = [0, 0]
        for mesh in meshes:
            for poly in mesh.geometry()[1]:
                count = len(poly)
                if count < 3 or count > 4:
                    raise Exception()
                primitives[len(poly) - 3] += 1

        triangles, quads = primitives[0] * 3, primitives[1] * 4
        length = triangles + quads
        self.vertices = numpy.zeros(length * 3, dtype=numpy.float32)
        self.normals = numpy.zeros(length * 3, dtype=numpy.float32)
        self.texels = numpy.zeros(length * 2, dtype=numpy.float32) if textured else None
        self.tangents = numpy.zeros(length * 3, dtype=numpy.float32) if textured else None

        if triangles > 0:
            self.parts.append(RenderMesh.Faceset(GL_TRIANGLES, 0, triangles))
        if quads > 0:
            self.parts.append(RenderMesh.Faceset(GL_QUADS, triangles, quads))

        try:
            smooth = appearance.smooth
        except:
            smooth = False

        # Initial positions
        index = [0, triangles]

        for mesh in meshes:
            geoVertices, geoPolygons = mesh.geometry()
            texVertices, texPolygons = mesh.texture()

            vertices = geoVertices if mesh.transform is None else [mesh.transform.apply(v) for v in geoVertices]

            def getNormal(points):
                return model.normalize(numpy.cross(
                        vertices[points[1]] - vertices[points[0]],
                        vertices[points[2]] - vertices[points[0]]))

            def getTangent(points, texels):
                return model.normalize(model.tangent(
                        vertices[points[1]] - vertices[points[0]],
                        vertices[points[2]] - vertices[points[0]],
                        texVertices[texels[1]] - texVertices[texels[0]],
                        texVertices[texels[2]] - texVertices[texels[0]]))

            if smooth:
                normals = [numpy.zeros(3) for i in range(0, len(geoVertices))]
                for poly in geoPolygons:
                    normal = getNormal(poly)
                    for vertex in poly:
                        normals[vertex] += normal
                normals = [model.normalize(vector) for vector in normals]

                if textured:
                    tangents = [numpy.zeros(3) for i in range(0, len(geoVertices))]
                    for gp, tp in zip(geoPolygons, texPolygons):
                        tangent = getTangent(gp, tp)
                        for vertex in gp:
                            tangents[vertex] += tangent
                    tangents = [model.normalize(vector) for vector in tangents]
            else:
                normals = [getNormal(gp) for gp in geoPolygons]
                if textured:
                    tangents = [getTangent(gp, tp) for gp, tp in zip(geoPolygons, texPolygons)]

            for i in range(0, len(geoPolygons)):
                gp = geoPolygons[i]
                if textured:
                    tp = texPolygons[i]

                count = len(gp)
                indexGroup = count - 3

                offset = index[indexGroup]
                index[indexGroup] += count

                for vertex in range(0, count):
                    geoStart, geoEnd, texStart, texEnd = 3 * offset, 3 * (offset + 1), 2 * offset, 2 * (offset + 1)

                    self.vertices[geoStart:geoEnd] = vertices[gp[vertex]][0:3]
                    self.normals[geoStart:geoEnd] = normals[gp[vertex]] if smooth else normals[i]
                    if textured:
                        self.texels[texStart:texEnd] = texVertices[tp[vertex]]
                        self.tangents[geoStart:geoEnd] = tangents[gp[vertex]] if smooth else tangents[i]
                    offset += 1

        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)

        self.verticesVBO = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.verticesVBO)
        glBufferData(GL_ARRAY_BUFFER, self.vertices, GL_STATIC_DRAW)
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, None)

        self.normalsVBO = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.normalsVBO)
        glBufferData(GL_ARRAY_BUFFER, self.normals, GL_STATIC_DRAW)
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, None)

        if textured:
            self.texelsVBO = glGenBuffers(1)
            glBindBuffer(GL_ARRAY_BUFFER, self.texelsVBO)
            glBufferData(GL_ARRAY_BUFFER, self.texels, GL_STATIC_DRAW)
            #glBufferData(GL_ARRAY_BUFFER, self.texels, GL_DYNAMIC_DRAW)
            glEnableVertexAttribArray(2);
            glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 0, None)

            self.tangentVBO = glGenBuffers(1)
            glBindBuffer(GL_ARRAY_BUFFER, self.tangentVBO)
            glBufferData(GL_ARRAY_BUFFER, self.tangents, GL_STATIC_DRAW)
            glEnableVertexAttribArray(3);
            glVertexAttribPointer(3, 3, GL_FLOAT, GL_FALSE, 0, None)

        glBindVertexArray(0)

        debug('Mesh created in {:f}, id {:s}, triangles {:d}, quads {:d}, vertices {:d}'.format(
                time.time() - started, self.ident, int(triangles / 3), int(quads / 4), length))

    def draw(self, projectionMatrix, modelViewMatrix, lights, wireframe):
        if self.appearance is not None:
            self.appearance.enable(projectionMatrix, modelViewMatrix, lights)
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
            glDrawArrays(entry.mode, entry.index, entry.count)
        glBindVertexArray(0)


class ScreenMesh(RenderMesh):
    def __init__(self, meshes):
        super().__init__(meshes)

    def draw(self):
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        glDisable(GL_CULL_FACE)

        glBindVertexArray(self.vao)
        for entry in self.parts:
            glDrawArrays(entry.mode, entry.index, entry.count)
        glBindVertexArray(0)


class Scene:
    class Camera:
        def __init__(self, fov=90.0, distance=20.0, translationRate=0.02, rotationRate=math.pi / 12.0, scalingRate=1.1):
            self.fov = fov
            self.distance = distance
            self.translationRate = translationRate
            self.rotationRate = rotationRate
            self.scalingRate = scalingRate
            self.reset()

        def reset(self):
            self.pov = numpy.array([0.0, 0.0, 0.0, 1.0])
            self.camera = numpy.array([0.0, -self.distance, 0.0, 1.0])
            self.axis = numpy.array([0.0, 0.0, 1.0, 0.0])

        def front(self):
            distance = numpy.linalg.norm(self.camera - self.pov)
            self.camera = numpy.array([0.0, -distance, 0.0, 0.0]) + self.pov
            self.axis = numpy.array([0.0, 0.0, 1.0, 0.0])

        def side(self):
            distance = numpy.linalg.norm(self.camera - self.pov)
            self.camera = numpy.array([distance, 0.0, 0.0, 0.0]) + self.pov
            self.axis = numpy.array([0.0, 0.0, 1.0, 0.0])

        def top(self):
            distance = numpy.linalg.norm(self.camera - self.pov)
            self.camera = numpy.array([0.0, 0.0, distance, 0.0]) + self.pov
            self.axis = numpy.array([0.0, 1.0, 0.0, 0.0])

        def rotate(self, hrot, vrot):
            camera = self.camera - self.pov
            axis = self.axis
            if hrot != 0.0:
                horizRotationMatrix = numpy.matrix([
                        [ math.cos(hrot), math.sin(hrot), 0.0, 0.0],
                        [-math.sin(hrot), math.cos(hrot), 0.0, 0.0],
                        [            0.0,            0.0, 1.0, 0.0],
                        [            0.0,            0.0, 0.0, 1.0]])
                camera = (camera * horizRotationMatrix).getA()[0]
                axis = (axis * horizRotationMatrix).getA()[0]
            if vrot != 0.0:
                normal = numpy.cross(camera[0:3], self.axis[0:3])
                normal /= numpy.linalg.norm(normal)
                vertRotationMatrix = model.rotationMatrix(normal, vrot).transpose()
                camera = (camera * vertRotationMatrix).getA()[0]
                axis = (axis * vertRotationMatrix).getA()[0]
            self.camera = camera + self.pov
            self.axis = axis

        def move(self, x, y):
            camera = self.camera - self.pov
            normal = numpy.cross(self.axis[0:3], camera[0:3])
            normal /= numpy.linalg.norm(normal)
            distance = numpy.linalg.norm(camera)
            width = 2.0 * math.tan(self.fov / 2.0) * distance

            offset = normal * (-x / width) + self.axis[0:3] * (-y / width)
            offset *= distance * self.translationRate
            offset = numpy.array([*offset, 0.0])

            self.camera += offset
            self.pov += offset

        def zoom(self, z):
            scale = numpy.array([z, z, z, 1.0])
            self.camera = (self.camera - self.pov) * scale + self.pov

        def zoomIn(self):
            self.zoom(1.0 / self.scalingRate)

        def zoomOut(self):
            self.zoom(self.scalingRate)


    def __init__(self):
        self.ortho = False
        self.depth = (0.001, 1000.0)
        self.camera = Scene.Camera()
        self.viewport = (640, 480)
        self.updateMatrix(self.viewport)

        self.lights = [
                [ 50.0,  50.0,  50.0, 1.0],
                [-50.0, -50.0, -50.0, 1.0]]

    def updateMatrix(self, viewport):
        aspect = float(viewport[0]) / float(viewport[1])
        if self.ortho:
            distance = numpy.linalg.norm(self.camera.camera - self.camera.pov)
            width = 1.0 / math.tan(self.camera.fov / 2.0) * distance
            area = (width, width / aspect)
            self.projectionMatrix = model.createOrthographicMatrix(area, self.depth)
        else:
            self.projectionMatrix = model.createPerspectiveMatrix(aspect, self.camera.fov, self.depth)

        self.modelViewMatrix = model.createModelViewMatrix(self.camera.camera[0:3], self.camera.pov[0:3],
                self.camera.axis[0:3])

    def resetShaders(self):
        glUseProgram(0)


class Shader:
    IDENT = 0

    def __init__(self):
        self.ident = Shader.IDENT
        Shader.IDENT += 1

        self.dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'shaders_ogl41')
        self.program = None

    def activateTexture(self, channel, texture):
        glActiveTexture(GL_TEXTURE0 + channel)
        glBindTexture(texture.mode, texture.buffer)
        if texture.mode == GL_TEXTURE_2D:
            glTexParameteri(texture.mode, GL_TEXTURE_WRAP_S, texture.repeatMode)
            glTexParameteri(texture.mode, GL_TEXTURE_WRAP_T, texture.repeatMode)
            glTexParameterf(texture.mode, GL_TEXTURE_MAG_FILTER, texture.filterMode[0])
            glTexParameterf(texture.mode, GL_TEXTURE_MIN_FILTER, texture.filterMode[1])
        glUniform1i(texture.location, channel)

    def create(self, vertex, fragment, geometry=None, control=None, evaluation=None):
        try:
            shaders = []
            shaders.append(compileShader(vertex, GL_VERTEX_SHADER))
            shaders.append(compileShader(fragment, GL_FRAGMENT_SHADER))
            if geometry is not None:
                shaders.append(compileShader(geometry, GL_GEOMETRY_SHADER))
            if control is not None:
                shaders.append(compileShader(control, GL_TESS_CONTROL_SHADER))
            if evaluation is not None:
                shaders.append(compileShader(evaluation, GL_TESS_EVALUATION_SHADER))
            self.program = compileProgram(*shaders)
            debug('Shader {:d} compiled'.format(self.ident))
        except RuntimeError as runError:
            print(runError.args[0]) # Print error log
            print('Shader {:d} compilation failed'.format(self.ident))
            exit()
        except:
            print('Unknown shader error')
            exit()

    def enable(self):
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

            def loadShaderFile(path):
                source = open(os.path.join(self.dir, path), 'rb').read().decode('utf-8').split('\n')
                source = [source[0]] + flags + source[1:]
                return '\n'.join(source)

            self.create(loadShaderFile('default.vert'), loadShaderFile('default.frag'))

        self.lightAmbient = 0.1
        self.lightDiffuse = numpy.array([[0.9, 0.9, 0.9], [0.9, 0.9, 0.9]], numpy.float32)

        self.projectionLoc = glGetUniformLocation(self.program, 'projectionMatrix')
        self.modelViewLoc = glGetUniformLocation(self.program, 'modelViewMatrix')
        self.normalLoc = glGetUniformLocation(self.program, 'normalMatrix')

        self.lightLoc = glGetUniformLocation(self.program, 'lightPosition')
        self.lightDiffuseLoc = glGetUniformLocation(self.program, 'lightDiffuseColor')
        self.lightAmbientLoc = glGetUniformLocation(self.program, 'lightAmbientIntensity')

    def enable(self, projectionMatrix, modelViewMatrix, lights):
        Shader.enable(self)

        # Set precalculated matrices
        normalMatrix = numpy.transpose(numpy.linalg.inv(modelViewMatrix))
        glUniformMatrix4fv(self.projectionLoc, 1, GL_FALSE, numpy.array(projectionMatrix, numpy.float32))
        glUniformMatrix4fv(self.modelViewLoc, 1, GL_FALSE, numpy.array(modelViewMatrix, numpy.float32))
        glUniformMatrix4fv(self.normalLoc, 1, GL_FALSE, numpy.array(normalMatrix, numpy.float32))

        # Set precalculated light positions, configure colors of diffuse and ambient lighting
        lights = numpy.array([(light * modelViewMatrix).getA()[0][0:3] for light in lights], numpy.float32)
        glUniform3fv(self.lightLoc, len(lights), lights)
        glUniform3fv(self.lightDiffuseLoc, 2, self.lightDiffuse)
        glUniform1f(self.lightAmbientLoc, self.lightAmbient)


class ModelShader(BaseModelShader):
    def __init__(self, texture=False, normal=False, specular=False):
        super().__init__(texture, normal, specular)

        self.diffuseColorLoc = glGetUniformLocation(self.program, 'materialDiffuseColor')
        self.specularColorLoc = glGetUniformLocation(self.program, 'materialSpecularColor')
        self.emissiveColorLoc = glGetUniformLocation(self.program, 'materialEmissiveColor')
        self.shininessLoc = glGetUniformLocation(self.program, 'materialShininess')

    def enable(self, projectionMatrix, modelViewMatrix, lights, colors, textures):
        BaseModelShader.enable(self, projectionMatrix, modelViewMatrix, lights)

        glUniform4fv(self.diffuseColorLoc, 1, list(colors.diffuse) + [1.0 - colors.transparency])
        glUniform3fv(self.specularColorLoc, 1, colors.specular)
        glUniform3fv(self.emissiveColorLoc, 1, colors.emissive)
        glUniform1f(self.shininessLoc, colors.shininess * 128.0)

        for i in range(0, len(textures)):
            self.activateTexture(i, textures[i])


class UnlitModelShader(Shader):
    def __init__(self):
        super().__init__()

        if self.program is None:
            loadShaderFile = lambda path: open(os.path.join(self.dir, path), 'rb').read().decode('utf-8')
            self.create(loadShaderFile('unlit.vert'), loadShaderFile('unlit.frag'))

        self.projectionLoc = glGetUniformLocation(self.program, 'projectionMatrix')
        self.modelViewLoc = glGetUniformLocation(self.program, 'modelViewMatrix')
        self.normalLoc = glGetUniformLocation(self.program, 'normalMatrix')

    def enable(self, projectionMatrix, modelViewMatrix, lights, colors, textures):
        Shader.enable(self)

        normalMatrix = numpy.transpose(numpy.linalg.inv(modelViewMatrix))
        glUniformMatrix4fv(self.projectionLoc, 1, GL_FALSE, numpy.array(projectionMatrix, numpy.float32))
        glUniformMatrix4fv(self.modelViewLoc, 1, GL_FALSE, numpy.array(modelViewMatrix, numpy.float32))
        glUniformMatrix4fv(self.normalLoc, 1, GL_FALSE, numpy.array(normalMatrix, numpy.float32))


class SystemShader(Shader):
    def __init__(self, prefix='', flags=[]):
        super().__init__()

        if self.program is None and prefix != '':
            def loadShaderFile(path, flags):
                source = open(os.path.join(self.dir, path), 'rb').read().decode('utf-8').split('\n')
                source = [source[0]] + flags + source[1:]
                return '\n'.join(source)

            self.create(loadShaderFile(prefix + '.vert', flags), loadShaderFile(prefix + '.frag', flags))

        self.projectionLoc = glGetUniformLocation(self.program, 'projectionMatrix')
        self.modelViewLoc = glGetUniformLocation(self.program, 'modelViewMatrix')

        self.projectionMatrix = model.createOrthographicMatrix(
                area=(1.0, 1.0),
                distance=(0.001, 1000.0))
        self.modelViewMatrix = model.createModelViewMatrix(
                eye=numpy.array([0.0, 0.0, 1.0]),
                center=numpy.zeros(3),
                up=numpy.array([0.0, 1.0, 0.0]))

    def enable(self):
        Shader.enable(self)

        glUniformMatrix4fv(self.projectionLoc, 1, GL_FALSE, numpy.array(self.projectionMatrix, numpy.float32))
        glUniformMatrix4fv(self.modelViewLoc, 1, GL_FALSE, numpy.array(self.modelViewMatrix, numpy.float32))


class BackgroundShader(SystemShader):
    def __init__(self):
        super().__init__('background')


class MergeShader(SystemShader):
    def __init__(self, antialiasing):
        flags = ['#define AA_SAMPLES ' + str(antialiasing)] if antialiasing > 0 else []
        super().__init__('merge', flags)

        mode = GL_TEXTURE_2D_MULTISAMPLE if antialiasing > 0 else GL_TEXTURE_2D
        self.colorTexture = Texture(mode, glGetUniformLocation(self.program, 'colorTexture'))

    def enable(self, colorBuffer):
        SystemShader.enable(self)

        self.colorTexture.buffer = colorBuffer
        self.activateTexture(0, self.colorTexture)
        if self.colorTexture.mode == GL_TEXTURE_2D:
            glGenerateMipmap(GL_TEXTURE_2D)


class BlurShader(SystemShader):

    def __init__(self, masked):
        flags = ['#define MASKED'] if masked else []
        super().__init__('blur', flags)

        self.colorTexture = Texture(mode=GL_TEXTURE_2D, location=glGetUniformLocation(self.program, 'colorTexture'))
        self.directionLoc = glGetUniformLocation(self.program, 'direction')

        if masked:
            self.maskTexture = Texture(mode=GL_TEXTURE_2D, location=glGetUniformLocation(self.program,
                    'maskTexture'), filtering=(GL_NEAREST, GL_NEAREST))
            self.sourceTexture = Texture(mode=GL_TEXTURE_2D, location=glGetUniformLocation(self.program,
                    'sourceTexture'))
        else:
            self.maskTexture = None
            self.sourceTexture = None

    def enable(self, resolution, direction, colorBuffer, sourceBuffer=None, maskBuffer=None):
        SystemShader.enable(self)

        glUniform2fv(self.directionLoc, 1, direction / resolution)

        self.colorTexture.buffer = colorBuffer
        self.activateTexture(0, self.colorTexture)

        if self.maskTexture is not None:
            self.maskTexture.buffer = maskBuffer
            self.activateTexture(1, self.maskTexture)
        if self.sourceTexture is not None:
            self.sourceTexture.buffer = sourceBuffer
            self.activateTexture(2, self.sourceTexture)


class Render(Scene):
    class Framebuffer:
        def __init__(self, size, antialiasing, depth):
            # Color buffer
            self.color = glGenTextures(1)

            if antialiasing > 0:
                glBindTexture(GL_TEXTURE_2D_MULTISAMPLE, self.color)
                glTexImage2DMultisample(GL_TEXTURE_2D_MULTISAMPLE, antialiasing, GL_RGBA8, size[0], size[1], GL_TRUE)
                glBindTexture(GL_TEXTURE_2D_MULTISAMPLE, 0)
            else:
                glBindTexture(GL_TEXTURE_2D, self.color)
                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, size[0], size[1], 0, GL_RGBA, GL_UNSIGNED_BYTE, None)
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
                glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D_MULTISAMPLE, self.color, 0)
            else:
                glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, self.color, 0)
            if depth:
                glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, self.depth)

            glBindFramebuffer(GL_FRAMEBUFFER, 0)

            debug('Framebuffer built, size {:d}x{:d}, color {:d}, depth {:d}'.format(
                    size[0], size[1], self.color, self.depth))

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


    class ShaderStorage:
        def __init__(self, antialiasing=0, overlay=False):
            self.shaders = {}
            self.background = None
            self.blur = None
            self.blurMasked = None
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

            if antialiasing > 0 or overlay:
                self.merge = MergeShader(antialiasing=antialiasing)

            if overlay:
                self.blur = BlurShader(masked=False)
                self.blurMasked = BlurShader(masked=True)


    def __init__(self, objects=[], options={}):
        super().__init__()

        self.parseOptions(options)

        self.cameraMove = False
        self.cameraRotate = False
        self.cameraCursor = [0.0, 0.0]

        self.data = []

        glutInit()
        glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE)
        glutInitWindowSize(self.viewport[0], self.viewport[1])
        self.titleText = 'OpenGL 4.1 render'
        self.window = glutCreateWindow(self.titleText)
        glutReshapeFunc(self.resize)
        glutDisplayFunc(self.drawScene)
        glutKeyboardFunc(self.keyHandler)
        glutMotionFunc(self.mouseMove)
        glutMouseFunc(self.mouseButton)

        self.initGraphics()
        self.initScene(objects)
        self.updateMatrix(self.viewport)

        glutMainLoop()

    def parseOptions(self, options):
        self.antialiasing = 0 if 'antialiasing' not in options else options['antialiasing']
        self.overlay = False if 'overlay' not in options else options['overlay']
        self.wireframe = False if 'wireframe' not in options else options['wireframe']
        if 'size' in options:
            self.viewport = options['size']
        self.useFramebuffers = self.antialiasing > 0 or self.overlay

    def initGraphics(self):
        glClearColor(0.0, 0.0, 0.0, 1.0)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        glClearDepth(1.0)
        glDepthFunc(GL_LESS)
        glDisable(GL_DEPTH_TEST)

        glEnable(GL_TEXTURE_2D)

        self.shaderStorage = Render.ShaderStorage(self.antialiasing, self.overlay)
        self.screenPlane = ScreenMesh([geometry.Plane((2.0, 2.0), (1, 1))])

        if self.useFramebuffers:
            self.framebuffers = []
            self.initFramebuffers()
        else:
            self.framebuffers = None

        if self.overlay:
            self.initOverlay(initial=True)

    def initFramebuffers(self):
        for fb in self.framebuffers:
            fb.free()
        self.framebuffers = []
        self.framebuffers.append(Render.Framebuffer(self.viewport, self.antialiasing, True))
        self.framebuffers.append(Render.Framebuffer(self.viewport, 0, False))
        self.framebuffers.append(Render.Framebuffer(self.viewport, 0, False))

    def initScene(self, objects):
        self.data = buildObjectGroups(self.shaderStorage.shaders, objects)

    def initOverlay(self, initial=False):
        # Buffer format is R5G5B5A1
        MASKED, UNMASKED = bytearray([0x21, 0x84]), bytearray([0xFE, 0xFF])
        PART = 0.25

        if not initial:
            self.overlayMask.free()
        self.overlayMask = None

        maskBuffer = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, maskBuffer)
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1)

        x, y = self.viewport
        maskData = MASKED * (x * int(y * PART)) + UNMASKED * (x * int(y * (1.0 - PART)))

        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB5_A1, x, y, 0, GL_RGBA, GL_UNSIGNED_SHORT_5_5_5_1, maskData)
        glBindTexture(GL_TEXTURE_2D, 0)
        self.overlayMask = Texture(mode=GL_TEXTURE_2D, location=0, identifier=maskBuffer)
        debug('Overlay built, size {:d}x{:d}, texture {:d}'.format(x, y, maskBuffer))

    def drawScene(self):
        # First pass
        if self.useFramebuffers:
            glBindFramebuffer(GL_FRAMEBUFFER, self.framebuffers[0].buffer)
            glDrawBuffers(1, [GL_COLOR_ATTACHMENT0])

            if self.antialiasing > 0:
                glEnable(GL_MULTISAMPLE)
        else:
            glBindFramebuffer(GL_FRAMEBUFFER, 0)

        # Enable writing to depth mask and clear it
        glDepthMask(GL_TRUE)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        self.resetShaders()

        # Draw background, do not use depth mask, depth test is disabled
        glDepthMask(GL_FALSE)
        self.shaderStorage.background.enable()
        self.screenPlane.draw()

        # Draw other objects, use depth mask and enable depth test
        glDepthMask(GL_TRUE)
        glEnable(GL_DEPTH_TEST)
        for renderObject in self.data:
            renderObject.draw(self.projectionMatrix, self.modelViewMatrix, self.lights, self.wireframe)
        glDisable(GL_DEPTH_TEST)

        # Second pass
        if self.useFramebuffers:
            if self.antialiasing > 0:
                glDisable(GL_MULTISAMPLE)

            # Do not use depth mask, depth test is disabled
            glDepthMask(GL_FALSE)

            if self.overlay:
                glBindFramebuffer(GL_FRAMEBUFFER, self.framebuffers[1].buffer)
                self.shaderStorage.merge.enable(self.framebuffers[0].color)
                self.screenPlane.draw()

                glBindFramebuffer(GL_FRAMEBUFFER, self.framebuffers[2].buffer)
                self.shaderStorage.blur.enable(self.viewport, numpy.array([0.0, 1.0]), self.framebuffers[1].color)
                self.screenPlane.draw()

                glBindFramebuffer(GL_FRAMEBUFFER, 0)
                self.shaderStorage.blurMasked.enable(self.viewport, numpy.array([1.0, 0.0]),
                        self.framebuffers[2].color, self.framebuffers[1].color, self.overlayMask.buffer)
                self.screenPlane.draw()
            else:
                glBindFramebuffer(GL_FRAMEBUFFER, 0)
                self.shaderStorage.merge.enable(self.framebuffers[0].color)
                self.screenPlane.draw()

        glutSwapBuffers()

    def resize(self, width, height):
        self.viewport = (width if width > 0 else 1, height if height > 0 else 1)
        self.updateMatrix(self.viewport)

        if self.useFramebuffers:
            self.initFramebuffers()
        if self.overlay:
            self.initOverlay()

        glViewport(0, 0, self.viewport[0], self.viewport[1])
        glutPostRedisplay()

    def mouseButton(self, bNumber, bAction, x, y):
        if bNumber == GLUT_LEFT_BUTTON:
            self.cameraRotate = bAction == GLUT_DOWN
            if self.cameraRotate:
                self.cameraCursor = [x, y]
        elif bNumber == GLUT_MIDDLE_BUTTON:
            self.cameraMove = bAction == GLUT_DOWN
            if self.cameraMove:
                self.cameraCursor = [x, y]
        elif bNumber == 3 and bAction == GLUT_DOWN:
            self.camera.zoomIn()
        elif bNumber == 4 and bAction == GLUT_DOWN:
            self.camera.zoomOut()
        self.updateMatrix(self.viewport)
        glutPostRedisplay()

    def mouseMove(self, x, y):
        if self.cameraRotate:
            hrot = (self.cameraCursor[0] - x) / 100.0
            vrot = (y - self.cameraCursor[1]) / 100.0
            self.camera.rotate(hrot, vrot)
            self.cameraCursor = [x, y]
        elif self.cameraMove:
            self.camera.move(x - self.cameraCursor[0], self.cameraCursor[1] - y)
            self.cameraCursor = [x, y]
        self.updateMatrix(self.viewport)
        glutPostRedisplay()

    def keyHandler(self, key, x, y):
        redisplay = True
        updated = True

        if key in (b'\x1B', b'q', b'Q'):
            exit()
        elif key in (b'1'):
            self.camera.front()
        elif key in (b'2'):
            self.camera.rotate(0.0, -self.camera.rotationRate)
        elif key in (b'3'):
            self.camera.side()
        elif key in (b'4'):
            self.camera.rotate(-self.camera.rotationRate, 0.0)
        elif key in (b'5'):
            self.ortho = not self.ortho
        elif key in (b'6'):
            self.camera.rotate(self.camera.rotationRate, 0.0)
        elif key in (b'7'):
            self.camera.top()
        elif key in (b'8'):
            self.camera.rotate(0.0, self.camera.rotationRate)
        elif key in (b'9'):
            self.camera.rotate(math.pi, 0.0)
        elif key in (b'.'):
            self.camera.reset()
        elif key in (b'-'):
            self.camera.zoomOut()
        elif key in (b'+'):
            self.camera.zoomIn()
        elif key in (b'z', b'Z'):
            self.wireframe = not self.wireframe
            updated = False
        else:
            redisplay = False
            updated = False

        if updated:
            self.updateMatrix(self.viewport)
        if redisplay:
            glutPostRedisplay()

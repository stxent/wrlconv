#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# render_ogl41.py
# Copyright (C) 2015 xent
# Project is distributed under the terms of the GNU General Public License v3.0

import math
import numpy
import os
import sys
import time

import model
import geometry

from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
from OpenGL.GL.shaders import *

try:
    from PIL import Image
    imagesEnabled = True
except:
    imagesEnabled = False
    print("Images disabled, please install imaging library")

debugEnabled = False

def debug(text):
    if debugEnabled:
        print(text)

def generateNormals(meshes):
    blueMaterial = model.Material()
    blueMaterial.color.diffuse = numpy.array([0., 0., 1.])
    normalLength = 0.2

    urchin = model.LineArray(name="Normals")
    urchin.visualAppearance.material = blueMaterial

    for mesh in meshes:
        if not mesh.appearance().normals:
            continue

        geoVertices, geoPolygons = mesh.geometry()
        smooth = mesh.appearance().smooth

        vertices = geoVertices if mesh.transform is None else [mesh.transform.process(v) for v in geoVertices]

        def getNormal(points):
            return model.normalize(model.normal(vertices[points[1]] - vertices[points[0]],\
                    vertices[points[2]] - vertices[points[0]]))

        if smooth:
            normals = [numpy.array([0., 0., 0.]) for i in range(0, len(geoVertices))]
            for poly in geoPolygons:
                normal = getNormal(poly)
                for vertex in poly:
                    normals[vertex] += normal
            normals = [model.normalize(vector) for vector in normals]
        else:
            normals = [getNormal(gp) for gp in geoPolygons]

        for i in range(0, len(geoPolygons)):
            gp = geoPolygons[i]
            position = numpy.array([0., 0., 0.])
            for vertex in gp:
                position += vertices[vertex]
            position /= float(len(gp))
            lastIndex = len(urchin.geoVertices)
            urchin.geoVertices.append(position)
            urchin.geoVertices.append(position + normals[i] * normalLength)
            urchin.geoPolygons.append([lastIndex, lastIndex + 1])

    return None if len(urchin.geoPolygons) == 0 else urchin

def buildObjectGroups(shaders, inputObjects):
    renderObjects = []
    objects = inputObjects

    #Render meshes
    meshes = filter(lambda entry: entry.style == model.Object.PATCHES, objects)

    normals = generateNormals(meshes)
    if normals is not None:
        inputObjects.append(normals)

    keys = []
    [keys.append(item) for item in map(lambda mesh: mesh.appearance().material, meshes) if item not in keys]

    groups = map(lambda key: filter(lambda mesh: mesh.appearance().material == key, objects), keys)
    [renderObjects.append(RenderMesh(shaders, group)) for group in groups]

    #Render line arrays
    arrays = filter(lambda entry: entry.style == model.Object.LINES, objects)
    if len(arrays) > 0:
        renderObjects.append(RenderLineArray(shaders, arrays))

    sortedObjects = filter(lambda entry: entry.appearance.material.color.transparency <= 0.001, renderObjects)
    sortedObjects += filter(lambda entry: entry.appearance.material.color.transparency > 0.001, renderObjects)

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
        debug("Overlay freed")


class RenderAppearance:
    class ImageTexture(Texture):
        def __init__(self, location, path):
            Texture.__init__(self, GL_TEXTURE_2D, location)

            if imagesEnabled:
                if not os.path.isfile(path[1]):
                    raise Exception()
                im = Image.open(path[1])
                try:
                    self.size, image = im.size, im.tostring("raw", "RGBA", 0, -1)
                except SystemError:
                    self.size, image = im.size, im.tostring("raw", "RGBX", 0, -1)
            else:
                self.size = (8, 8)
                pBlack, pPink = "\x00\x00\x00\xFF", "\xFF\x00\xFF\xFF"
                width, height = self.size[0] / 2, self.size[1] / 2
                image = ((pBlack + pPink) * width + (pPink + pBlack) * width) * height
                self.filterMode = (GL_NEAREST, GL_NEAREST)

            self.buffer = glGenTextures(1)

            glBindTexture(self.mode, self.buffer)
            glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
            glTexImage2D(self.mode, 0, GL_RGBA8, self.size[0], self.size[1], 0, GL_RGBA, GL_UNSIGNED_BYTE, image)
            debug("Texture loaded: %s, width: %d, height: %d, id: %d"\
                    % (path[0], self.size[0], self.size[1], self.buffer))


    def __init__(self, shaders, appearance):
        self.textures = []
        self.shader = None

        self.smooth = False
        self.solid = False
        self.wireframe = False
        self.zbuffer = True

        if appearance is None:
            self.material = model.Material()
            self.shader = shaders["Unlit"]
        else:
            self.material = appearance.material
            self.smooth = appearance.smooth
            self.wireframe = appearance.wireframe
            self.solid = False if appearance.wireframe else appearance.solid

            shaderName = ""
            if self.material.diffuse is not None:
                shaderName += "Diff"
            if self.material.normal is not None:
                shaderName += "Norm"
            if self.material.specular is not None:
                shaderName += "Spec"

            if shaderName == "":
                shaderName = "Colored"

            self.shader = shaders[shaderName]

        if self.material.diffuse is not None:
            self.textures.append(RenderAppearance.ImageTexture(glGetUniformLocation(self.shader.program,\
                    "diffuseTexture"), self.material.diffuse.path))
        if self.material.normal is not None:
            self.textures.append(RenderAppearance.ImageTexture(glGetUniformLocation(self.shader.program,\
                    "normalTexture"), self.material.normal.path))
        if self.material.specular is not None:
            self.textures.append(RenderAppearance.ImageTexture(glGetUniformLocation(self.shader.program,\
                    "specularTexture"), self.material.specular.path))

    def enable(self, scene):
        if self.shader is not None:
            scene.enableShader(self.shader, [scene, self.material.color])
            for i in range(0, len(self.textures)):
                self.shader.activateTexture(i, self.textures[i])


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

    def draw(self, wireframe=False):
        pass


class RenderLineArray(RenderObject):
    def __init__(self, shaders, meshes):
        RenderObject.__init__(self)

        self.parts = []
        self.appearance = RenderAppearance(shaders, None)

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

        #Initial positions
        index = [0]

        for mesh in meshes:
            geoVertices, geoPolygons = mesh.geometry()
            color = mesh.appearance().material.color.diffuse

            vertices = geoVertices if mesh.transform is None else [mesh.transform.process(v) for v in geoVertices]

            for gp in geoPolygons:
                count = len(gp)
                indexGroup = count - 2

                offset = index[indexGroup]
                index[indexGroup] += count

                for vertex in range(0, count):
                    start, end = 3 * offset, 3 * (offset + 1)
                    self.vertices[start:end] = numpy.array(numpy.swapaxes(vertices[gp[vertex]][0:3], 0, 1))
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

        debug("Point cloud created in %f, id %s, lines %u, vertices %u"\
                % (time.time() - started, self.ident, lines / 2, length))

    def draw(self, wireframe=False):
        glBindVertexArray(self.vao)
        for entry in self.parts:
            glDrawArrays(entry.mode, entry.index, entry.count)
        glBindVertexArray(0)


class RenderMesh(RenderObject):
    def __init__(self, shaders, meshes):
        RenderObject.__init__(self)

        self.parts = []
        self.appearance = RenderAppearance(shaders, meshes[0].appearance())

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

        #Initial positions
        index = [0, triangles]
        smooth = self.appearance.smooth

        for mesh in meshes:
            geoVertices, geoPolygons = mesh.geometry()
            texVertices, texPolygons = mesh.texture()

            vertices = geoVertices if mesh.transform is None else [mesh.transform.process(v) for v in geoVertices]

            def getNormal(points):
                return model.normalize(model.normal(vertices[points[1]] - vertices[points[0]],\
                        vertices[points[2]] - vertices[points[0]]))

            def getTangent(points, texels):
                return model.normalize(model.tangent(vertices[points[1]] - vertices[points[0]],\
                        vertices[points[2]] - vertices[points[0]],\
                        texVertices[texels[1]] - texVertices[texels[0]],\
                        texVertices[texels[2]] - texVertices[texels[0]]))

            if smooth:
                normals = [numpy.array([0., 0., 0.]) for i in range(0, len(geoVertices))]
                for poly in geoPolygons:
                    normal = getNormal(poly)
                    for vertex in poly:
                        normals[vertex] += normal
                normals = [model.normalize(vector) for vector in normals]

                if textured:
                    tangents = [numpy.array([0., 0., 0.]) for i in range(0, len(geoVertices))]
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

                    self.vertices[geoStart:geoEnd] = numpy.array(numpy.swapaxes(vertices[gp[vertex]][0:3], 0, 1))
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

        debug("Mesh created in %f, id %s, triangles %u, quads %u, vertices %u"\
                % (time.time() - started, self.ident, triangles / 3, quads / 4, length))

    def draw(self, wireframe=False):
        if wireframe or self.appearance.wireframe:
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
            glDisable(GL_CULL_FACE)
        else:
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
            if self.appearance.solid:
                glEnable(GL_CULL_FACE)
                glCullFace(GL_BACK)
            else:
                glDisable(GL_CULL_FACE)

        glBindVertexArray(self.vao)
        for entry in self.parts:
            glDrawArrays(entry.mode, entry.index, entry.count)
        glBindVertexArray(0)


class Scene:
    class Camera:
        DISTANCE = 20.

        def __init__(self):
            self.fov = 90.
            self.home()

        def home(self):
            self.pov = numpy.matrix([[0.], [0.], [0.], [1.]])
            self.camera = numpy.matrix([[0.], [-Render.Camera.DISTANCE], [0.], [1.]])
            self.axis = numpy.matrix([[0.], [0.], [1.], [0.]])

        def front(self):
            self.camera -= self.pov
            distance = numpy.linalg.norm(numpy.array(self.camera[:,0][0:3]))
            self.camera = numpy.matrix([[0.], [-distance], [0.], [0.]])
            self.axis = numpy.matrix([[0.], [0.], [1.], [0.]])
            self.camera += self.pov

        def side(self):
            self.camera -= self.pov
            distance = numpy.linalg.norm(numpy.array(self.camera[:,0][0:3]))
            self.camera = numpy.matrix([[distance], [0.], [0.], [0.]])
            self.axis = numpy.matrix([[0.], [0.], [1.], [0.]])
            self.camera += self.pov

        def top(self):
            self.camera -= self.pov
            distance = numpy.linalg.norm(numpy.array(self.camera[:,0][0:3]))
            self.camera = numpy.matrix([[0.], [0.], [distance], [0.]])
            self.axis = numpy.matrix([[0.], [1.], [0.], [0.]])
            self.camera += self.pov

        def rotate(self, hrot, vrot):
            self.camera -= self.pov
            if hrot != 0.:
                horizRotationMatrix = numpy.matrix([
                        [math.cos(hrot), -math.sin(hrot), 0., 0.],
                        [math.sin(hrot),  math.cos(hrot), 0., 0.],
                        [            0.,              0., 1., 0.],
                        [            0.,              0., 0., 1.]])
                self.camera = horizRotationMatrix * self.camera
                self.axis = horizRotationMatrix * self.axis
            if vrot != 0.:
                normal = model.normal(self.camera, self.axis)
                normal = model.normalize(normal)
                vertRotationMatrix = model.rotationMatrix(normal, vrot)
                self.camera = vertRotationMatrix * self.camera
                self.axis = vertRotationMatrix * self.axis
            self.camera += self.pov

        def move(self, x, y):
            self.camera -= self.pov
            axis = numpy.array(self.axis)[:,0][0:3]
            camera = numpy.array(self.camera)[:,0][0:3]
            normal = model.normalize(model.normal(axis, camera))

            width = 2. * math.tan(self.fov / 2.) * numpy.linalg.norm(camera)

            offset = normal * (-x / width) + axis * (-y / width)
            rotationMatrix = numpy.matrix([
                    [1., 0., 0., offset[0]],
                    [0., 1., 0., offset[1]],
                    [0., 0., 1., offset[2]],
                    [0., 0., 0.,        1.]])
            self.camera += self.pov
            self.camera = rotationMatrix * self.camera
            self.pov = rotationMatrix * self.pov

        def zoom(self, z):
            scaleMatrix = numpy.matrix([
                    [ z, 0., 0., 0.],
                    [0.,  z, 0., 0.],
                    [0., 0.,  z, 0.],
                    [0., 0., 0., 1.]])
            self.camera -= self.pov
            self.camera = scaleMatrix * self.camera
            self.camera += self.pov


    def __init__(self):
        self.ortho = False
        self.depth = (0.1, 1000.)
        self.camera = Scene.Camera()
        self.viewport = (640, 480)
        self.updateMatrix(self.viewport)
        self.lights = numpy.array([[50., 50., 50.], [-50., -50., -50.]])
        self.shader = None
        self.shaders = [] #TODO Rename

    def updateMatrix(self, viewport):
        aspect = float(viewport[0]) / float(viewport[1])
        if self.ortho:
            distance = numpy.linalg.norm(self.camera.camera - self.camera.pov)
            width = 1. / math.tan(self.camera.fov / 2.) * distance
            area = (width, width / aspect)
            self.projectionMatrix = model.createOrthographicMatrix(area, self.depth)
        else:
            self.projectionMatrix = model.createPerspectiveMatrix(aspect, self.camera.fov, self.depth)

        self.modelViewMatrix = model.createModelViewMatrix(self.camera.camera, self.camera.pov, self.camera.axis)
        self.normalMatrix = numpy.transpose(numpy.linalg.inv(self.modelViewMatrix))

    def enableShader(self, shader, arguments):
        shader.enable(*arguments)
        self.shader = shader.ident

    def resetShaders(self):
        glUseProgram(0)
        self.shader = None


class Shader:
    IDENT = 0

    def __init__(self):
        self.dir = "./shaders/"
        self.ident = Shader.IDENT
        Shader.IDENT += 1
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

    def enable(self):
        glUseProgram(self.program)

    def create(self, vertex, fragment):
        try:
            vertexShader = compileShader(vertex, GL_VERTEX_SHADER)
            fragmentShader = compileShader(fragment, GL_FRAGMENT_SHADER)
            self.program = compileProgram(vertexShader, fragmentShader)
            debug("Shader %u compiled" % self.ident)
        except RuntimeError as runError:
            print(runError.args[0]) #Print error log
            print("Shader %u compilation failed" % self.ident)
            exit()
        except:
            print("Unknown shader error")
            exit()


class UnlitShader(Shader):
    def __init__(self):
        Shader.__init__(self)

        if self.program is None:
            vert, frag = map(lambda path: open(self.dir + path, "rb").read(), ["unlit.vert", "unlit.frag"])
            self.create(vert, frag)

        self.projectionLoc = glGetUniformLocation(self.program, "projectionMatrix")
        self.modelViewLoc = glGetUniformLocation(self.program, "modelViewMatrix")
        self.normalLoc = glGetUniformLocation(self.program, "normalMatrix")

    def enable(self, scene, color):
        Shader.enable(self)

        glUniformMatrix4fv(self.projectionLoc, 1, GL_FALSE, numpy.array(scene.projectionMatrix, numpy.float32))
        glUniformMatrix4fv(self.modelViewLoc, 1, GL_FALSE, numpy.array(scene.modelViewMatrix, numpy.float32))
        glUniformMatrix4fv(self.normalLoc, 1, GL_FALSE, numpy.array(scene.normalMatrix, numpy.float32))


class ModelShader(Shader):
    def __init__(self, texture, normal, specular):
        Shader.__init__(self)

        if self.program is None:
            flags = []
            flags += ["#define LIGHT_COUNT 2"]
            if texture:
                flags += ["#define DIFFUSE_MAP"]
            if normal:
                flags += ["#define NORMAL_MAP"]
            if specular:
                flags += ["#define SPECULAR_MAP"]

            code = map(lambda path: open(self.dir + path, "rb").read(), ["default.vert", "default.frag"])
            code = map(lambda text: text.split("\n"), code)
            code = map(lambda text: [text[0]] + flags + text[1:], code)
            vert, frag = map("\n".join, code)
            self.create(vert, frag)

        self.projectionLoc = glGetUniformLocation(self.program, "projectionMatrix")
        self.modelViewLoc = glGetUniformLocation(self.program, "modelViewMatrix")
        self.normalLoc = glGetUniformLocation(self.program, "normalMatrix")

        self.lightLoc = glGetUniformLocation(self.program, "lightPosition")
        self.lightDiffuseLoc = glGetUniformLocation(self.program, "lightDiffuseColor")
        self.lightAmbientLoc = glGetUniformLocation(self.program, "lightAmbientIntensity")

    def enable(self, scene):
        Shader.enable(self)

        glUniformMatrix4fv(self.projectionLoc, 1, GL_FALSE, numpy.array(scene.projectionMatrix, numpy.float32))
        glUniformMatrix4fv(self.modelViewLoc, 1, GL_FALSE, numpy.array(scene.modelViewMatrix, numpy.float32))
        glUniformMatrix4fv(self.normalLoc, 1, GL_FALSE, numpy.array(scene.normalMatrix, numpy.float32))

        ambient = 0.1
        diffuse = numpy.array([[0.9, 0.9, 0.9], [0.9, 0.9, 0.9]], numpy.float32)

        glUniform3fv(self.lightLoc, len(scene.lights), numpy.array(scene.lights, numpy.float32))
        glUniform3fv(self.lightDiffuseLoc, 2, diffuse)
        glUniform1f(self.lightAmbientLoc, ambient)


class DefaultShader(ModelShader):
    def __init__(self, texture=False, normal=False, specular=False):
        ModelShader.__init__(self, texture, normal, specular)

        self.diffuseColorLoc = glGetUniformLocation(self.program, "materialDiffuseColor")
        self.specularColorLoc = glGetUniformLocation(self.program, "materialSpecularColor")
        self.emissiveColorLoc = glGetUniformLocation(self.program, "materialEmissiveColor")
        self.shininessLoc = glGetUniformLocation(self.program, "materialShininess")

    def enable(self, scene, color):
        ModelShader.enable(self, scene)

        glUniform4fv(self.diffuseColorLoc, 1, list(color.diffuse) + [1. - color.transparency])
        glUniform3fv(self.specularColorLoc, 1, color.specular)
        glUniform3fv(self.emissiveColorLoc, 1, color.emissive)
        glUniform1f(self.shininessLoc, color.shininess * 128.0)


class BackgroundShader(Shader):
    def __init__(self):
        Shader.__init__(self)

        if self.program is None:
            vert, frag = map(lambda path: open(self.dir + path, "rb").read(),\
                    ["background.vert", "background.frag"])
            self.create(vert, frag)

        self.projectionLoc = glGetUniformLocation(self.program, "projectionMatrix")
        self.modelViewLoc = glGetUniformLocation(self.program, "modelViewMatrix")

        camera = numpy.matrix([[0.], [0.], [1.], [0.]])
        axis = numpy.matrix([[0.], [1.], [0.], [0.]])
        pov = numpy.matrix([[0.], [0.], [0.], [1.]])

        self.projectionMatrix = model.createOrthographicMatrix((1.0, 1.0), (0.001, 1000.0))
        self.modelViewMatrix = model.createModelViewMatrix(camera, pov, axis)

    def enable(self):
        Shader.enable(self)

        glUniformMatrix4fv(self.projectionLoc, 1, GL_FALSE, numpy.array(self.projectionMatrix, numpy.float32))
        glUniformMatrix4fv(self.modelViewLoc, 1, GL_FALSE, numpy.array(self.modelViewMatrix, numpy.float32))


class MergeShader(Shader):
    def __init__(self, antialiasing):
        Shader.__init__(self)

        self.antialiasing = antialiasing

        if self.program is None:
            flags = []
            if self.antialiasing > 0:
                flags += ["#define AA_SAMPLES %u" % antialiasing]

            code = map(lambda path: open(self.dir + path, "rb").read(), ["merge.vert", "merge.frag"])
            code = map(lambda text: text.split("\n"), code)
            code = map(lambda text: [text[0]] + flags + text[1:], code)
            vert, frag = map("\n".join, code)
            self.create(vert, frag)

        self.projectionLoc = glGetUniformLocation(self.program, "projectionMatrix")
        self.modelViewLoc = glGetUniformLocation(self.program, "modelViewMatrix")

        camera = numpy.matrix([[0.], [0.], [1.], [0.]])
        axis = numpy.matrix([[0.], [1.], [0.], [0.]])
        pov = numpy.matrix([[0.], [0.], [0.], [1.]])

        self.projectionMatrix = model.createOrthographicMatrix((1.0, 1.0), (0.001, 1000.0))
        self.modelViewMatrix = model.createModelViewMatrix(camera, pov, axis)

        mode = GL_TEXTURE_2D_MULTISAMPLE if self.antialiasing > 0 else GL_TEXTURE_2D
        self.colorTexture = Texture(mode, glGetUniformLocation(self.program, "colorTexture"))

    def enable(self, colorBuffer):
        Shader.enable(self)

        glUniformMatrix4fv(self.projectionLoc, 1, GL_FALSE, numpy.array(self.projectionMatrix, numpy.float32))
        glUniformMatrix4fv(self.modelViewLoc, 1, GL_FALSE, numpy.array(self.modelViewMatrix, numpy.float32))

        self.colorTexture.buffer = colorBuffer
        self.activateTexture(0, self.colorTexture)
        if self.colorTexture.mode == GL_TEXTURE_2D:
            glGenerateMipmap(GL_TEXTURE_2D)


class BlurShader(Shader):
    def __init__(self, masked):
        Shader.__init__(self)

        self.masked = masked

        if self.program is None:
            flags = []
            if self.masked:
                flags += ["#define MASKED"]

            code = map(lambda path: open(self.dir + path, "rb").read(), ["blur.vert", "blur.frag"])
            code = map(lambda text: text.split("\n"), code)
            code = map(lambda text: [text[0]] + flags + text[1:], code)
            vert, frag = map("\n".join, code)
            self.create(vert, frag)

        self.projectionLoc = glGetUniformLocation(self.program, "projectionMatrix")
        self.modelViewLoc = glGetUniformLocation(self.program, "modelViewMatrix")

        camera = numpy.matrix([[0.], [0.], [1.], [0.]])
        axis = numpy.matrix([[0.], [1.], [0.], [0.]])
        pov = numpy.matrix([[0.], [0.], [0.], [1.]])

        self.projectionMatrix = model.createOrthographicMatrix((1.0, 1.0), (0.001, 1000.0))
        self.modelViewMatrix = model.createModelViewMatrix(camera, pov, axis)

        self.colorTexture = Texture(mode=GL_TEXTURE_2D, location=glGetUniformLocation(self.program, "colorTexture"))
        self.directionLoc = glGetUniformLocation(self.program, "direction")

        if self.masked:
            self.maskTexture = Texture(mode=GL_TEXTURE_2D, location=glGetUniformLocation(self.program, "maskTexture"),\
                    filtering=(GL_NEAREST, GL_NEAREST))
            self.sourceTexture = Texture(mode=GL_TEXTURE_2D, location=glGetUniformLocation(self.program,\
                    "sourceTexture"))

    def enable(self, scene, direction, colorBuffer, sourceBuffer=None, maskBuffer=None):
        Shader.enable(self)

        glUniformMatrix4fv(self.projectionLoc, 1, GL_FALSE, numpy.array(self.projectionMatrix, numpy.float32))
        glUniformMatrix4fv(self.modelViewLoc, 1, GL_FALSE, numpy.array(self.modelViewMatrix, numpy.float32))
        glUniform2fv(self.directionLoc, 1, direction / scene.viewport)

        self.colorTexture.buffer = colorBuffer
        self.activateTexture(0, self.colorTexture)

        if self.masked:
            self.maskTexture.buffer = maskBuffer
            self.activateTexture(1, self.maskTexture)
            self.sourceTexture.buffer = sourceBuffer
            self.activateTexture(2, self.sourceTexture)


class Render(Scene):
    class Framebuffer:
        def __init__(self, size, antialiasing, depth):
            #Color buffer
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
                #Depth buffer
                self.depth = glGenRenderbuffers(1)

                if antialiasing > 0:
                    glBindRenderbuffer(GL_RENDERBUFFER, self.depth)
                    glRenderbufferStorageMultisample(GL_RENDERBUFFER, antialiasing, GL_DEPTH_COMPONENT,\
                            size[0], size[1])
                    glBindRenderbuffer(GL_RENDERBUFFER, 0)
                else:
                    glBindRenderbuffer(GL_RENDERBUFFER, self.depth)
                    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, size[0], size[1])
                    glBindRenderbuffer(GL_RENDERBUFFER, 0)
            else:
                self.depth = 0

            #Create framebuffer
            self.buffer = glGenFramebuffers(1)
            glBindFramebuffer(GL_FRAMEBUFFER, self.buffer)

            if antialiasing > 0:
                glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D_MULTISAMPLE, self.color, 0)
            else:
                glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, self.color, 0)
            if depth:
                glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, self.depth)

            glBindFramebuffer(GL_FRAMEBUFFER, 0)

            debug("Framebuffer built, size %ux%u, color %u, depth %u" % (size[0], size[1], self.color, self.depth))

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

            debug("Framebuffer freed")


    class ShaderStorage:
        def __init__(self, antialiasing=0, overlay=False):
            self.shaders = {}
            self.background = None
            self.blur = None
            self.blurMasked = None
            self.merge = None

            oldDir = os.getcwd()
            scriptDir = os.path.dirname(os.path.realpath(__file__))
            if len(scriptDir) > 0:
                os.chdir(scriptDir)

            self.background = BackgroundShader()

            self.shaders["Colored"] = DefaultShader()
            self.shaders["Unlit"] = UnlitShader()

            self.shaders["Diff"] = DefaultShader(texture=True, normal=False, specular=False)
            self.shaders["Norm"] = DefaultShader(texture=False, normal=True, specular=False)
            self.shaders["Spec"] = DefaultShader(texture=False, normal=False, specular=True)
            self.shaders["DiffNorm"] = DefaultShader(texture=True, normal=True, specular=False)
            self.shaders["DiffSpec"] = DefaultShader(texture=True, normal=False, specular=True)
            self.shaders["NormSpec"] = DefaultShader(texture=False, normal=True, specular=True)
            self.shaders["DiffNormSpec"] = DefaultShader(texture=True, normal=True, specular=True)

            if antialiasing > 0 or overlay:
                self.merge = MergeShader(antialiasing=antialiasing)

            if overlay:
                self.blur = BlurShader(masked=False)
                self.blurMasked = BlurShader(masked=True)

            os.chdir(oldDir)


    def __init__(self, objects=[], options={}):
        Scene.__init__(self)

        self.parseOptions(options)

        self.cameraMove = False
        self.cameraRotate = False
        self.cameraCursor = [0., 0.]

        self.data = []

        glutInit(sys.argv)
        glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE)
        glutInitWindowSize(self.viewport[0], self.viewport[1])
        self.titleText = "OpenGL 4.1 render"
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
        self.antialiasing = 0 if "antialiasing" not in options.keys() else options["antialiasing"]
        self.overlay = False if "overlay" not in options.keys() else options["overlay"]
        self.wireframe = False if "wireframe" not in options.keys() else options["wireframe"]
        if "size" in options.keys():
            self.viewport = options["size"]
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
        self.mergePlane = RenderMesh(self.shaderStorage.shaders, [geometry.Plane((2., 2.), (1, 1))])

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
        if not initial:
            self.overlayMask.free()
        self.overlayMask = None

        maskBuffer = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, maskBuffer)
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1)

        part = 32
        image = "\xFF" * (self.viewport[0] * part) + "\x00" * (self.viewport[0] * (self.viewport[1] - part))

        glTexImage2D(GL_TEXTURE_2D, 0, GL_R8, self.viewport[0], self.viewport[1], 0, GL_RED, GL_UNSIGNED_BYTE, image)
        glBindTexture(GL_TEXTURE_2D, 0)
        self.overlayMask = Texture(mode=GL_TEXTURE_2D, location=0, identifier=maskBuffer)
        debug("Overlay built, size %ux%u, texture %u" % (self.viewport[0], self.viewport[1], maskBuffer))

    def drawScene(self):
        #First pass
        if self.useFramebuffers:
            glBindFramebuffer(GL_FRAMEBUFFER, self.framebuffers[0].buffer)
            glDrawBuffers(1, [GL_COLOR_ATTACHMENT0])

            if self.antialiasing > 0:
                glEnable(GL_MULTISAMPLE)
        else:
            glBindFramebuffer(GL_FRAMEBUFFER, 0)

        #Enable writing to depth mask and clear it
        glDepthMask(GL_TRUE)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        self.resetShaders()

        #Draw background, do not use depth mask, depth test is disabled
        glDepthMask(GL_FALSE)
        self.enableShader(self.shaderStorage.background, [])
        self.mergePlane.draw()

        #Draw other objects, use depth mask and enable depth test
        glDepthMask(GL_TRUE)
        glEnable(GL_DEPTH_TEST)
        for current in self.data:
            current.appearance.enable(self)
            current.draw(self.wireframe)
        glDisable(GL_DEPTH_TEST)

        #Second pass
        if self.useFramebuffers:
            if self.antialiasing > 0:
                glDisable(GL_MULTISAMPLE)

            #Do not use depth mask, depth test is disabled
            glDepthMask(GL_FALSE)

            if self.overlay:
                glBindFramebuffer(GL_FRAMEBUFFER, self.framebuffers[1].buffer)
                self.enableShader(self.shaderStorage.merge, [self.framebuffers[0].color])
                self.mergePlane.draw()

                glBindFramebuffer(GL_FRAMEBUFFER, self.framebuffers[2].buffer)
                self.enableShader(self.shaderStorage.blur, [self, numpy.array([0., 1.]), self.framebuffers[1].color])
                self.mergePlane.draw()

                glBindFramebuffer(GL_FRAMEBUFFER, 0)
                self.enableShader(self.shaderStorage.blurMasked, [self, numpy.array([1., 0.]),
                        self.framebuffers[2].color, self.framebuffers[1].color, self.overlayMask.buffer])
                self.mergePlane.draw()
            else:
                glBindFramebuffer(GL_FRAMEBUFFER, 0)
                self.enableShader(self.shaderStorage.merge, [self.framebuffers[0].color])
                self.mergePlane.draw()

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
            self.camera.zoom(0.9)
        elif bNumber == 4 and bAction == GLUT_DOWN:
            self.camera.zoom(1.1)
        self.updateMatrix(self.viewport)
        glutPostRedisplay()

    def mouseMove(self, x, y):
        if self.cameraRotate:
            hrot = (self.cameraCursor[0] - x) / 100.
            vrot = (y - self.cameraCursor[1]) / 100.
            self.camera.rotate(hrot, vrot)
            self.cameraCursor = [x, y]
        elif self.cameraMove:
            self.camera.move(x - self.cameraCursor[0], self.cameraCursor[1] - y)
            self.cameraCursor = [x, y]
        self.updateMatrix(self.viewport)
        glutPostRedisplay()

    def keyHandler(self, key, x, y):
        updated = False

        if key in ("\x1b", "q", "Q"):
            exit()
        elif key in ("1"):
            self.camera.front()
            updated = True
        elif key in ("3"):
            self.camera.side()
            updated = True
        elif key in ("5"):
            self.ortho = not self.ortho
            updated = True
        elif key in ("7"):
            self.camera.top()
            updated = True
        elif key in ("."):
            self.camera.home()
            updated = True
        elif key in ("z", "Z"):
            self.wireframe = not self.wireframe

        if updated:
            self.updateMatrix(self.viewport)
        glutPostRedisplay()

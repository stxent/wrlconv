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

try:
    from OpenGL.GL import *
    from OpenGL.GLU import *
    from OpenGL.GLUT import *
    from OpenGL.GL.shaders import *
    from PIL import Image
except:
    exit()

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

def buildObjectGroups(inputObjects):
    data = []
    objects = inputObjects

    #Render meshes
    meshes = filter(lambda entry: entry.style == model.Object.PATCHES, objects)

    normals = generateNormals(meshes)
    if normals is not None:
        inputObjects.append(normals)

    keys = []
    [keys.append(item) for item in map(lambda mesh: mesh.appearance().material, meshes) if item not in keys]

    groups = map(lambda key: filter(lambda mesh: mesh.appearance().material == key, objects), keys)
    [data.append(RenderMesh(group)) for group in groups]

    #Render line arrays
    arrays = filter(lambda entry: entry.style == model.Object.LINES, objects)
    if len(arrays) > 0:
        data.append(RenderLineArray(arrays))

    return data


class RenderAppearance:
    class Texture:
        def __init__(self, path):
            if not os.path.isfile(path[1]):
                raise Exception()
            im = Image.open(path[1])
            try:
                self.size, image = im.size, im.tostring("raw", "RGBA", 0, -1)
            except SystemError:
                self.size, image = im.size, im.tostring("raw", "RGBX", 0, -1)

            self.buf = glGenTextures(1)
#            self.kind = GL_TEXTURE_RECTANGLE
            self.kind = GL_TEXTURE_2D

            glBindTexture(self.kind, self.buf)
            glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
#            glTexImage2D(self.kind, 0, 3, self.size[0], self.size[1], 0, GL_RGBA, GL_UNSIGNED_BYTE, image)
            gluBuild2DMipmaps(self.kind, 4, self.size[0], self.size[1], GL_RGBA, GL_UNSIGNED_BYTE, image)
            debug("Texture loaded: %s, width: %d, height: %d, id: %d"\
                    % (path[0], self.size[0], self.size[1], self.buf))


    def __init__(self, appearance):
        self.textures = []
        self.zbuffer = True

        if appearance is None:
            self.name = "Unlit"
            self.solid = False
        else:
            self.material = appearance.material
            self.smooth = appearance.smooth
            self.solid = appearance.solid
            self.wireframe = appearance.wireframe

            name = ""
            if self.material.diffuse is not None:
                name += "Diff"
                self.textures.append(RenderAppearance.Texture(self.material.diffuse.path))
            if self.material.normal is not None:
                name += "Norm"
                self.textures.append(RenderAppearance.Texture(self.material.normal.path))
            if self.material.specular is not None:
                name += "Spec"
                self.textures.append(RenderAppearance.Texture(self.material.specular.path))

            self.name = name if name != "" else "Colored"

    def enable(self, scene):
        scene.shaders[self.name].enable(self)

        for i in range(0, len(self.textures)):
            glActiveTexture(GL_TEXTURE0 + i)
            glEnable(self.textures[i].kind)
            glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
            glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
            glBindTexture(self.textures[i].kind, self.textures[i].buf)
        if self.solid:
            glEnable(GL_CULL_FACE)
            glCullFace(GL_BACK)
        else:
            glDisable(GL_CULL_FACE)


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
    def __init__(self, meshes):
        RenderObject.__init__(self)

        self.parts = []
        self.appearance = RenderAppearance(None)

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
    def __init__(self, meshes):
        RenderObject.__init__(self)

        self.parts = []
        self.appearance = RenderAppearance(meshes[0].appearance())

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
        else:
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)

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
        self.updateMatrix((320, 240))
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

    def resetShaders(self):
        glUseProgram(0)
        self.shader = None
        glDisable(GL_TEXTURE_2D)
        glDisable(GL_TEXTURE_RECTANGLE)


class Shader:
    IDENT = 0

    def __init__(self, name, scene):
        self.name = name
        self.scene = scene
        self.ident = Shader.IDENT
        Shader.IDENT += 1
        self.program = None

    def enable(self, view):
        glUseProgram(self.program)
        self.scene.shader = self.ident

    def create(self, vertex, fragment):
        try:
            vertexShader = compileShader(vertex, GL_VERTEX_SHADER)
            fragmentShader = compileShader(fragment, GL_FRAGMENT_SHADER)
            self.program = compileProgram(vertexShader, fragmentShader)
        except RuntimeError as runError:
            print(runError.args[0]) #Print error log
            print("Shader compilation failed")
            exit()
        except:
            print("Unknown shader error")
            exit()


class UnlitShader(Shader):
    def __init__(self, name, scene):
        Shader.__init__(self, name, scene)

        if self.program is None:
            vert, frag = map(lambda path: open(path, "rb").read(), ["./shaders/unlit.vert", "./shaders/unlit.frag"])
            self.create(vert, frag)

        self.projectionLoc = glGetUniformLocation(self.program, "projectionMatrix")
        self.modelViewLoc = glGetUniformLocation(self.program, "modelViewMatrix")
        self.normalLoc = glGetUniformLocation(self.program, "normalMatrix")

    def enable(self, view):
        if self.scene.shader == self.ident:
            return
        Shader.enable(self, view)

        glUniformMatrix4fv(self.projectionLoc, 1, GL_FALSE, numpy.array(self.scene.projectionMatrix, numpy.float32))
        glUniformMatrix4fv(self.modelViewLoc, 1, GL_FALSE, numpy.array(self.scene.modelViewMatrix, numpy.float32))
        glUniformMatrix4fv(self.normalLoc, 1, GL_FALSE, numpy.array(self.scene.normalMatrix, numpy.float32))


class ModelShader(Shader):
    def __init__(self, name, scene, texture, normal, specular):
        Shader.__init__(self, name, scene)

        if self.program is None:
            flags = ""
            flags += "#define LIGHT_COUNT 2\n"
            if texture:
                flags += "#define DIFFUSE_MAP\n"
            if normal:
                flags += "#define NORMAL_MAP\n"
            if specular:
                flags += "#define SPECULAR_MAP\n"

            vert, frag = map(lambda path: flags + open(path, "rb").read(),\
                    ["./shaders/default.vert", "./shaders/default.frag"])
            self.create(vert, frag)

        self.projectionLoc = glGetUniformLocation(self.program, "projectionMatrix")
        self.modelViewLoc = glGetUniformLocation(self.program, "modelViewMatrix")
        self.normalLoc = glGetUniformLocation(self.program, "normalMatrix")

        self.lightLoc = glGetUniformLocation(self.program, "lightPosition")
        self.lightDiffuseLoc = glGetUniformLocation(self.program, "lightDiffuseColor")
        self.lightAmbientLoc = glGetUniformLocation(self.program, "lightAmbientIntensity")

    def enable(self, view):
        if self.scene.shader == self.ident:
            return
        Shader.enable(self, view)

        glUniformMatrix4fv(self.projectionLoc, 1, GL_FALSE, numpy.array(self.scene.projectionMatrix, numpy.float32))
        glUniformMatrix4fv(self.modelViewLoc, 1, GL_FALSE, numpy.array(self.scene.modelViewMatrix, numpy.float32))
        glUniformMatrix4fv(self.normalLoc, 1, GL_FALSE, numpy.array(self.scene.normalMatrix, numpy.float32))

        ambient = 0.1
        diffuse = numpy.array([[0.9, 0.9, 0.9], [0.9, 0.9, 0.9]], numpy.float32)

        glUniform3fv(self.lightLoc, len(self.scene.lights), numpy.array(self.scene.lights, numpy.float32))
        glUniform3fv(self.lightDiffuseLoc, 2, diffuse)
        glUniform1f(self.lightAmbientLoc, ambient)


class DefaultShader(ModelShader):
    def __init__(self, name, scene, texture=False, normal=False, specular=False):
        ModelShader.__init__(self, name, scene, texture, normal, specular)
        self.diffuseColorLoc = glGetUniformLocation(self.program, "materialDiffuseColor")
        self.specularColorLoc = glGetUniformLocation(self.program, "materialSpecularColor")
        self.emissiveColorLoc = glGetUniformLocation(self.program, "materialEmissiveColor")
        self.shininessLoc = glGetUniformLocation(self.program, "materialShininess")

        self.texLoc = []
        if texture:
            self.texLoc.append(glGetUniformLocation(self.program, "diffuseTexture"))
        if normal:
            self.texLoc.append(glGetUniformLocation(self.program, "normalTexture"))
        if specular:
            self.texLoc.append(glGetUniformLocation(self.program, "specularTexture"))

    def enable(self, view):
        ModelShader.enable(self, view)

        color = view.material.color
        glUniform4fv(self.diffuseColorLoc, 1, list(color.diffuse) + [1. - color.transparency])
        glUniform3fv(self.specularColorLoc, 1, color.specular)
        glUniform3fv(self.emissiveColorLoc, 1, color.emissive)
        glUniform1f(self.shininessLoc, color.shininess * 128.0)

        for i in range(0, len(self.texLoc)):
            glUniform1i(self.texLoc[i], i)


class Render(Scene):
    def __init__(self, objects=[]):
        Scene.__init__(self)

        #self.updated = True
        self.cameraMove = False
        self.cameraRotate = False
        self.cameraCursor = [0., 0.]

        self.viewport = (640, 480)
        self.wireframe = False
        self.frames = 0
        self.data = []

        glutInit(sys.argv)
        glutInitDisplayMode(GLUT_RGBA)
        glutInitWindowSize(self.viewport[0], self.viewport[1])
        self.window = glutCreateWindow("OpenGL 4.1 render")
        glutReshapeFunc(self.resize)
        glutDisplayFunc(self.drawScene)
        glutIdleFunc(self.idleScene)
        glutKeyboardFunc(self.keyHandler)
        glutMotionFunc(self.mouseMove)
        glutMouseFunc(self.mouseButton)

        self.initGraphics()
        self.initScene(objects)

        self.updateMatrix(self.viewport)

        glutMainLoop()

    def initGraphics(self):
        self.loadShaders()
        glClearColor(0.0, 0.0, 0.0, 1.0)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glClearDepth(1.0)
        glDepthFunc(GL_LESS)
        glEnable(GL_DEPTH_TEST)

    def loadShaders(self):
        oldDir = os.getcwd()
        scriptDir = os.path.dirname(os.path.realpath(__file__))
        if len(scriptDir) > 0:
            os.chdir(scriptDir)
        self.shaders = {}
        self.shaders["Colored"] = DefaultShader(name="Colored", scene=self)
        self.shaders["Unlit"] = UnlitShader("Unlit", self)

        self.shaders["Diff"] = DefaultShader(name="Diff", scene=self,\
                texture=True, normal=False, specular=False)
        self.shaders["Norm"] = DefaultShader(name="Norm", scene=self,\
                texture=False, normal=True, specular=False)
        self.shaders["Spec"] = DefaultShader(name="Spec", scene=self,\
                texture=False, normal=False, specular=True)
        self.shaders["DiffNorm"] = DefaultShader(name="DiffNorm", scene=self,\
                texture=True, normal=True, specular=False)
        self.shaders["DiffSpec"] = DefaultShader(name="DiffSpec", scene=self,\
                texture=True, normal=False, specular=True)
        self.shaders["NormSpec"] = DefaultShader(name="NormSpec", scene=self,\
                texture=False, normal=True, specular=True)
        self.shaders["DiffNormSpec"] = DefaultShader(name="DiffNormSpec", scene=self,\
                texture=True, normal=True, specular=True)

        os.chdir(oldDir)

    def initScene(self, objects):
        self.data = buildObjectGroups(objects)

    def idleScene(self):
        self.drawScene() #FIXME
        time.sleep(0.005)

    def drawScene(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        self.resetShaders()
        for current in self.data:
            current.appearance.enable(self)
            current.draw(self.wireframe)

        glutSwapBuffers()
        self.frames += 1

    def resize(self, width, height):
        self.viewport = (width if width > 0 else 1, height if height > 0 else 1)
        self.updateMatrix(self.viewport)
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

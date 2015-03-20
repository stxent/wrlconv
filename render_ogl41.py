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
import Image

import model

try:
    from OpenGL.GL import *
    from OpenGL.GLU import *
    from OpenGL.GLUT import *
    from OpenGL.GL.shaders import *
except:
    exit()

debugEnabled = False

def debug(text):
    if debugEnabled:
        print(text)


class RenderAppearance:
    class Texture:
        def __init__(self, path):
            if not os.path.isfile(path):
                raise Exception()
            im = Image.open(path)
            try:
                self.size, image = im.size, im.tostring("raw", "RGBA", 0, -1)
            except SystemError:
                self.size, image = im.size, im.tostring("raw", "RGBX", 0, -1)

            self.buf = glGenTextures(1)
            self.kind = GL_TEXTURE_RECTANGLE
            #self.type = GL_TEXTURE_2D

            glBindTexture(self.kind, self.buf)
            glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
            glTexImage2D(self.kind, 0, 3, self.size[0], self.size[1], 0, GL_RGBA, GL_UNSIGNED_BYTE, image)
            debug("Texture loaded: %s, width: %d, height: %d, id: %d" % (path, self.size[0], self.size[1], self.buf))

    def __init__(self, material):
        self.material = material
        self.textures = []

        self.name = ""
        if material.diffuse is None:
            self.name = "colored"
        else:
            self.name = "textured"
            self.textures.append(RenderAppearance.Texture(material.diffuse.path))

    def enable(self, scene):
        scene.shaders[self.name].enable(self)
        for i in range(0, len(self.textures)):
            glActiveTexture(GL_TEXTURE0 + i)
            glEnable(self.textures[i].kind)
            #glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            #glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
            #glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
            glBindTexture(self.textures[i].kind, self.textures[i].buf)


class RenderMesh:
    IDENT = 0

    class Faceset:
        def __init__(self, mode, index, count):
            self.mode = mode
            self.index = index
            self.count = count

    def __init__(self, meshes):
        self.ident = str(RenderMesh.IDENT)
        RenderMesh.IDENT += 1

        self.parts = []
        self.smooth = meshes[0].smooth
        self.zbuffer = True

        self.solid = True
        self.debug = False #Show normals
        self.appearance = RenderAppearance(meshes[0].material)

        textured = len(meshes[0].texPolygons) != 0
        started = time.time()

        polys = []
        for mesh in meshes:
            polys.extend(map(lambda polygon: len(polygon), mesh.geoPolygons))
        triangles, quads = len(filter(lambda x: x == 3, polys)), len(filter(lambda x: x == 4, polys))

        length = triangles * 3 + quads * 4
        self.vertices = numpy.zeros(length * 3, dtype=numpy.float32)
        self.normals = numpy.zeros(length * 3, dtype=numpy.float32)
        self.texels = numpy.zeros(length * 2, dtype=numpy.float32) if textured else None
        self.tangents = numpy.zeros(length * 3, dtype=numpy.float32) if textured else None

        if triangles > 0:
            self.parts.append(RenderMesh.Faceset(GL_TRIANGLES, 0, triangles * 3))
        if quads > 0:
            self.parts.append(RenderMesh.Faceset(GL_QUADS, triangles * 3, quads * 4))

        #TODO Add define
        index = [0, triangles * 3] #Initial positions for triangle and quad samples

        for mesh in meshes:
            transformed = []
            if mesh.transform is not None:
                transformed = map(lambda x: mesh.transform.process(x), mesh.geoVertices)
            else:
                transformed = mesh.geoVertices

            vertexIndex = 0
            for i in range(0, len(mesh.geoPolygons)):
                poly = mesh.geoPolygons[i]

                normal = model.normalize(model.normal(transformed[poly[1]] - transformed[poly[0]],\
                        transformed[poly[2]] - transformed[poly[0]]))
                if textured:
                    tangent = model.normalize(model.tangent(\
                            transformed[poly[1]] - transformed[poly[0]],\
                            transformed[poly[2]] - transformed[poly[0]],\
                            mesh.texVertices[vertexIndex + 1] - mesh.texVertices[vertexIndex],\
                            mesh.texVertices[vertexIndex + 2] - mesh.texVertices[vertexIndex]))

                current = 0 if len(poly) == 3 else 1
                offset = index[current]
                index[current] += len(poly)

                for j in range(0, len(poly)):
                    vertex = transformed[poly[j]]
                    self.vertices[3 * offset:3 * offset + 3] = numpy.array(numpy.swapaxes(vertex[0:3], 0, 1))
                    self.normals[3 * offset:3 * offset + 3] = normal
                    if textured:
                        self.texels[2 * offset:2 * offset + 2] = mesh.texVertices[vertexIndex]
                        self.tangents[3 * offset:3 * offset + 3] = tangent
                    offset += 1
                    vertexIndex += 1

        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)

        self.verticesVBO = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.verticesVBO)
        glBufferData(GL_ARRAY_BUFFER, self.vertices, GL_STATIC_DRAW)
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, None);

        self.normalsVBO = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.normalsVBO)
        glBufferData(GL_ARRAY_BUFFER, self.normals, GL_STATIC_DRAW)
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, None);

        if textured:
            self.texelsVBO = glGenBuffers(1)
            glBindBuffer(GL_ARRAY_BUFFER, self.texelsVBO)
            glBufferData(GL_ARRAY_BUFFER, self.texels, GL_STATIC_DRAW)
            #glBufferData(GL_ARRAY_BUFFER, self.texels, GL_DYNAMIC_DRAW)
            glEnableVertexAttribArray(2);
            glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 0, None);

        glBindVertexArray(0)

        debug("Mesh created in %f, id %s, triangles %u, quads %u, vertices %u"\
                % (time.time() - started, self.ident, triangles, quads, len(self.vertices) / 3))

    def draw(self, wireframe=False):
        if wireframe:
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)

        glBindVertexArray(self.vao)
        for entry in self.parts:
            glDrawArrays(entry.mode, entry.index, entry.count)
        glBindVertexArray(0)

        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)


class Scene:
    def __init__(self):
        self.updateModelView(numpy.array([1.0, 0.0, 0.0]), numpy.array([0.0, 0.0, 0.0]), numpy.array([0.0, 0.0, 1.0]))
        self.updateProjection((640, 480))
        self.lights = numpy.array([[50.0, 50.0, 50.0], [-50.0, -50.0, -50.0]])
        self.shader = None
        self.shaders = [] #TODO Rename

    def updateModelView(self, eye, center, up):
        self.modelViewMatrix = model.createModelViewMatrix(eye, center, up)
        self.normalMatrix = self.modelViewMatrix
        #self.normalMatrix = numpy.delete(numpy.delete(self.normalMatrix, 3, 1), 3, 0)
        self.normalMatrix = numpy.transpose(numpy.linalg.inv(self.normalMatrix))

    def updateProjection(self, size, angle = 45.0):
        self.projectionMatrix = model.createPerspectiveMatrix(size, angle, (0.1, 1000.0))

    def resetShaders(self):
        glUseProgram(0)
        self.shader = None
        glDisable(GL_TEXTURE_2D)
        glDisable(GL_TEXTURE_RECTANGLE)


class Shader:
    IDENT = 0

    def __init__(self, name, scene):
        self.ident = Shader.IDENT
        self.scene = scene
        Shader.IDENT += 1
        self.name = name
        if not hasattr(self, 'program'): #FIXME Rewrite
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


class ModelShader(Shader):
    def __init__(self, name, scene):
        Shader.__init__(self, name, scene)

        if self.program is None:
            content = []
            for path in ["./shaders/%s.vert", "./shaders/%s.frag"]:
                desc = open(path % name, "rb")
                content.append(desc.read())
                desc.close()
            self.create(content[0], content[1])

        self.projectionLoc = glGetUniformLocation(self.program, "projectionMatrix")
        self.modelViewLoc = glGetUniformLocation(self.program, "modelViewMatrix")
        self.normalLoc = glGetUniformLocation(self.program, "normalMatrix")

        self.lightLoc = glGetUniformLocation(self.program, "lightPosition")
        self.lightDiffuseLoc = glGetUniformLocation(self.program, "lightDiffuseColor")
        self.lightAmbientLoc = glGetUniformLocation(self.program, "lightAmbientIntensity")

    def enable(self, view):
        Shader.enable(self, view)

        glUniformMatrix4fv(self.projectionLoc, 1, GL_FALSE, numpy.array(self.scene.projectionMatrix, numpy.float32))
        glUniformMatrix4fv(self.modelViewLoc, 1, GL_FALSE, numpy.array(self.scene.modelViewMatrix, numpy.float32))
        glUniformMatrix4fv(self.normalLoc, 1, GL_FALSE, numpy.array(self.scene.normalMatrix, numpy.float32))

        diffuse = numpy.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]], numpy.float32)
        glUniform3fv(self.lightLoc, len(self.scene.lights), numpy.array(self.scene.lights, numpy.float32))
        glUniform3fv(self.lightDiffuseLoc, 2, diffuse)
        glUniform1f(self.lightAmbientLoc, 0.0)


class ColorShader(ModelShader):
    def __init__(self, name, scene):
        ModelShader.__init__(self, name, scene)
        self.diffuseColorLoc = glGetUniformLocation(self.program, "materialDiffuseColor")
        self.specularColorLoc = glGetUniformLocation(self.program, "materialSpecularColor")
        self.emissiveColorLoc = glGetUniformLocation(self.program, "materialEmissiveColor")
        self.shininessLoc = glGetUniformLocation(self.program, "materialShininess")

    def enable(self, view):
        loaded = self.scene.shader == self.ident
        if not loaded:
            ModelShader.enable(self, view)
        diffuse = [view.material.color.diffuse[0], view.material.color.diffuse[1], view.material.color.diffuse[2],\
                1. - view.material.color.transparency]
        glUniform4fv(self.diffuseColorLoc, 1, diffuse)
        glUniform3fv(self.specularColorLoc, 1, view.material.color.specular)
        glUniform3fv(self.emissiveColorLoc, 1, view.material.color.emissive)
        glUniform1f(self.shininessLoc, view.material.color.shininess * 128.0)


class TextureShader(ModelShader):
    def __init__(self, name, scene):
        ModelShader.__init__(self, name, scene)
        self.diffuseColorLoc = glGetUniformLocation(self.program, "materialDiffuseColor")
        self.specularColorLoc = glGetUniformLocation(self.program, "materialSpecularColor")
        self.emissiveColorLoc = glGetUniformLocation(self.program, "materialEmissiveColor")
        self.shininessLoc = glGetUniformLocation(self.program, "materialShininess")
        self.textureLoc = glGetUniformLocation(self.program, "diffuseTexture")

    def enable(self, view):
        loaded = self.scene.shader == self.ident
        if not loaded:
            ModelShader.enable(self, view)
        glUniform4fv(self.diffuseColorLoc, 1, view.material.color.diffuse)
        glUniform3fv(self.specularColorLoc, 1, view.material.color.specular)
        glUniform3fv(self.emissiveColorLoc, 1, view.material.color.emissive)
        glUniform1f(self.shininessLoc, view.material.color.shininess * 128.0)
        glUniform1i(self.textureLoc, 0)


class Render(Scene):
    def __init__(self, objects=[]):
        Scene.__init__(self)

        self.camera = numpy.matrix([[0.], [20.], [20.], [1.]])
        self.pov    = numpy.matrix([[0.], [0.], [0.], [1.]])
        self.axis   = numpy.matrix([[0.], [0.], [1.], [1.]])
        #self.updated = True
        self.rotateCamera = False
        self.moveCamera = False
        self.mousePos = [0., 0.]
        self.viewport = (640, 480)
        self.wireframe = False
        self.frames = 0
        self.data = []

        glutInit(sys.argv)
        glutInitDisplayMode(GLUT_RGBA)
        glutInitWindowSize(self.viewport[0], self.viewport[1])
        self.window = glutCreateWindow("OpenGL 4.2 render")
        glutReshapeFunc(self.resize)
        glutDisplayFunc(self.drawScene)
        glutIdleFunc(self.idleScene)
        glutKeyboardFunc(self.keyHandler)
        glutMotionFunc(self.mouseMove)
        glutMouseFunc(self.mouseButton)

        self.initGraphics()
        self.initScene(objects)

        self.updateProjection(self.viewport)
        self.updateModelView(self.camera, self.pov, self.axis)

        glutMainLoop()

    def initGraphics(self):
        #glClearDepth(1.0)
        #glDepthFunc(GL_LESS)
        #glEnable(GL_DEPTH_TEST)
        ##Blending using shader
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        ##glCullFace(GL_BACK)
        self.loadShaders()
        glEnable(GL_DEPTH_TEST)
        #glDisable(GL_CULL_FACE)
        glCullFace(GL_BACK)
        glClearColor(0.0, 0.0, 0.0, 1.0)

    def loadShaders(self):
        oldDir = os.getcwd()
        scriptDir = os.path.dirname(os.path.realpath(__file__))
        if len(scriptDir) > 0:
            os.chdir(scriptDir)
        self.shaders = {}
        self.shaders["colored"] = ColorShader("colored", self)
        self.shaders["textured"] = TextureShader("textured", self)
        os.chdir(oldDir)

    def initScene(self, objects):
        materials = []
        [materials.append(item) for item in map(lambda x: x.material, objects) if item not in materials]

        groups = map(lambda key: filter(lambda mesh: mesh.material == key, objects), materials)
        [self.data.append(RenderMesh(meshes)) for meshes in groups]

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
        self.updateProjection(self.viewport)
        glViewport(0, 0, self.viewport[0], self.viewport[1])
        glutPostRedisplay()

    def mouseButton(self, bNumber, bAction, xPos, yPos):
        if bNumber == GLUT_LEFT_BUTTON:
            if bAction == GLUT_DOWN:
                self.rotateCamera = True
                self.mousePos = [xPos, yPos]
            else:
                self.rotateCamera = False
        elif bNumber == GLUT_MIDDLE_BUTTON:
            if bAction == GLUT_DOWN:
                self.moveCamera = True
                self.mousePos = [xPos, yPos]
            else:
                self.moveCamera = False
        elif bNumber == 3 and bAction == GLUT_DOWN:
            zm = 0.9
            scaleMatrix = numpy.matrix([
                    [zm, 0., 0., 0.],
                    [0., zm, 0., 0.],
                    [0., 0., zm, 0.],
                    [0., 0., 0., 1.]])
            self.camera -= self.pov
            self.camera = scaleMatrix * self.camera
            self.camera += self.pov
        elif bNumber == 4 and bAction == GLUT_DOWN:
            zm = 1.1
            scaleMatrix = numpy.matrix([
                    [zm, 0., 0., 0.],
                    [0., zm, 0., 0.],
                    [0., 0., zm, 0.],
                    [0., 0., 0., 1.]])
            self.camera -= self.pov
            self.camera = scaleMatrix * self.camera
            self.camera += self.pov
        self.updateModelView(self.camera, self.pov, self.axis)
        glutPostRedisplay()

    def mouseMove(self, xPos, yPos):
        if self.rotateCamera:
            self.camera -= self.pov
            zrot = (self.mousePos[0] - xPos) / 100.
            nrot = (yPos - self.mousePos[1]) / 100.
            if zrot != 0.:
                rotMatrixA = numpy.matrix([
                        [math.cos(zrot), -math.sin(zrot), 0., 0.],
                        [math.sin(zrot),  math.cos(zrot), 0., 0.],
                        [            0.,              0., 1., 0.],
                        [            0.,              0., 0., 1.]])
                self.camera = rotMatrixA * self.camera
            if nrot != 0.:
                normal = model.normalize(model.normal(self.camera, self.axis))
                angle = model.angle(self.camera, self.axis)
                if (nrot > 0 and nrot > angle) or (nrot < 0 and -nrot > math.pi - angle):
                    self.axis = -self.axis
                rotMatrixB = model.rotationMatrix(normal, nrot)
                self.camera = rotMatrixB * self.camera
            self.camera += self.pov
            self.mousePos = [xPos, yPos]
        elif self.moveCamera:
            tlVector = numpy.matrix([[(xPos - self.mousePos[0]) / 50.], [(self.mousePos[1] - yPos) / 50.], [0.], [0.]])
            self.camera -= self.pov
            normal = model.normalize(model.normal([0., 0., 1.], self.camera))
            angle = model.angle(self.camera, [0., 0., 1.])
            ah = model.angle(normal, [1., 0., 0.])
            rotZ = numpy.matrix([
                    [math.cos(ah), -math.sin(ah), 0., 0.],
                    [math.sin(ah),  math.cos(ah), 0., 0.],
                    [          0.,            0., 1., 0.],
                    [          0.,            0., 0., 1.]])
            self.camera += self.pov
            rotCNormal = model.rotationMatrix(normal, angle)
            tlVector = rotZ * tlVector
            tlVector = rotCNormal * tlVector
            self.camera = self.camera - tlVector
            self.pov = self.pov - tlVector
            self.mousePos = [xPos, yPos]
        self.updateModelView(self.camera, self.pov, self.axis)
        glutPostRedisplay()

    def keyHandler(self, key, xPos, yPos):
        if key in ("\x1b", "q", "Q"):
            exit()
        elif key in ("r", "R"):
            self.camera = numpy.matrix([[0.], [20.], [0.], [1.]])
            self.pov = numpy.matrix([[0.], [0.], [0.], [1.]])
        elif key in ("w", "W"):
            self.wireframe = not self.wireframe
        self.updateModelView(self.camera, self.pov, self.axis)
        glutPostRedisplay()

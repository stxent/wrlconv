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
        else:
            self.material = appearance.material
            self.smooth = appearance.smooth
            self.solid = appearance.solid
            self.wireframe = appearance.wireframe

            if self.material.diffuse is None and self.material.normalmap is None:
                self.name = "Colored"
            elif self.material.diffuse is not None and self.material.normalmap is None:
                self.name = "Textured"
                self.textures.append(RenderAppearance.Texture(self.material.diffuse.path))
            elif self.material.diffuse is None and self.material.normalmap is not None:
                self.name = "ColoredBump"
                self.textures.append(RenderAppearance.Texture(self.material.normalmap.path))
            elif self.material.diffuse is not None and self.material.normalmap is not None:
                self.name = "TexturedBump"
                self.textures.append(RenderAppearance.Texture(self.material.diffuse.path))
                self.textures.append(RenderAppearance.Texture(self.material.normalmap.path))
            else:
                raise Exception()

    def enable(self, scene):
        scene.shaders[self.name].enable(self)

        for i in range(0, len(self.textures)):
            glActiveTexture(GL_TEXTURE0 + i)
            glEnable(self.textures[i].kind)
            glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
            glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
            glBindTexture(self.textures[i].kind, self.textures[i].buf)


class RenderMesh:
    class Faceset:
        def __init__(self, mode, index, count):
            self.mode = mode
            self.index = index
            self.count = count


    IDENT = 0

    def __init__(self, meshes):
        self.ident = str(RenderMesh.IDENT)
        RenderMesh.IDENT += 1

        self.parts = []
        self.appearance = RenderAppearance(meshes[0].appearance())

        textured = meshes[0].isTextured()
        started = time.time()

        triangles, quads = 0, 0
        for mesh in meshes:
            for poly in mesh.geometry()[1]:
                if len(poly) == 3:
                    triangles += 1
                else:
                    quads += 1

        length = triangles * 3 + quads * 4
        self.vertices = numpy.zeros(length * 3, dtype=numpy.float32)
        self.normals = numpy.zeros(length * 3, dtype=numpy.float32)
        self.texels = numpy.zeros(length * 2, dtype=numpy.float32) if textured else None
        self.tangents = numpy.zeros(length * 3, dtype=numpy.float32) if textured else None

        if triangles > 0:
            self.parts.append(RenderMesh.Faceset(GL_TRIANGLES, 0, triangles * 3))
        if quads > 0:
            self.parts.append(RenderMesh.Faceset(GL_QUADS, triangles * 3, quads * 4))

        #Initial positions for triangle and quad samples
        index = [0, triangles * 3]
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

                if count == 3:
                    indexGroup = 0
                elif count == 4:
                    indexGroup = 1
                else:
                    continue

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
                % (time.time() - started, self.ident, triangles, quads, len(self.vertices) / 3))

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
    def __init__(self):
        self.updateModelView(\
                numpy.matrix([[1.], [0.], [0.], [1.]]),\
                numpy.matrix([[0.], [0.], [0.], [1.]]),\
                numpy.matrix([[0.], [0.], [1.], [1.]]))
        self.updateProjection((640, 480))
        self.lights = numpy.array([[50., 50., 50.], [-50., -50., -50.]])
        self.shader = None
        self.shaders = [] #TODO Rename

    def updateModelView(self, eye, center, up):
        self.modelViewMatrix = model.createModelViewMatrix(eye, center, up)
        self.normalMatrix = numpy.transpose(numpy.linalg.inv(self.modelViewMatrix))

    def updateProjection(self, size, angle=45.):
        self.projectionMatrix = model.createPerspectiveMatrix(size, angle, (0.1, 1000.))

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

            content = []
            for path in ["./shaders/default.vert", "./shaders/default.frag"]:
                desc = open(path, "rb")
                content.append(flags + desc.read())
                desc.close()
            self.create(content[0], content[1])

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

    def enable(self, view):
        ModelShader.enable(self, view)

        color = view.material.color
        glUniform4fv(self.diffuseColorLoc, 1, list(color.diffuse) + [1. - color.transparency])
        glUniform3fv(self.specularColorLoc, 1, color.specular)
        glUniform3fv(self.emissiveColorLoc, 1, color.emissive)
        glUniform1f(self.shininessLoc, color.shininess * 128.0)


class ColoredBumpShader(DefaultShader):
    def __init__(self, name, scene):
        DefaultShader.__init__(self, name, scene, False, True, False)
        self.normalMapLoc = glGetUniformLocation(self.program, "normalTexture")

    def enable(self, view):
        DefaultShader.enable(self, view)

        glUniform1i(self.normalMapLoc, 0)


class TexturedShader(DefaultShader):
    def __init__(self, name, scene):
        DefaultShader.__init__(self, name, scene, True, False, False)
        self.diffuseMapLoc = glGetUniformLocation(self.program, "diffuseTexture")

    def enable(self, view):
        DefaultShader.enable(self, view)

        glUniform1i(self.diffuseMapLoc, 0)


class TexturedBumpShader(DefaultShader):
    def __init__(self, name, scene):
        DefaultShader.__init__(self, name, scene, True, True, False)
        self.diffuseMapLoc = glGetUniformLocation(self.program, "diffuseTexture")
        self.normalMapLoc = glGetUniformLocation(self.program, "normalTexture")

    def enable(self, view):
        DefaultShader.enable(self, view)

        glUniform1i(self.diffuseMapLoc, 0)
        glUniform1i(self.normalMapLoc, 1)


class Render(Scene):
    class Camera:
        DISTANCE = 20.

        def __init__(self):
            self.home()
            self.fov = 45.

        def home(self):
            self.pov = numpy.matrix([[0.], [0.], [0.], [1.]])
            self.camera = numpy.matrix([[Render.Camera.DISTANCE], [0.], [0.], [1.]])
            self.axis = numpy.matrix([[0.], [0.], [1.], [0.]])

        def front(self):
            self.camera -= self.pov
            distance = numpy.linalg.norm(numpy.array(self.camera[:,0][0:3]))
            self.camera = numpy.matrix([[distance], [0.], [0.], [0.]])
            self.axis = numpy.matrix([[0.], [0.], [1.], [0.]])
            self.camera += self.pov

        def side(self):
            self.camera -= self.pov
            distance = numpy.linalg.norm(numpy.array(self.camera[:,0][0:3]))
            self.camera = numpy.matrix([[0.], [-distance], [0.], [0.]])
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


    def __init__(self, objects=[]):
        Scene.__init__(self)

        #self.updated = True
        self.camera = Render.Camera()
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
        self.updateModelView(self.camera.camera, self.camera.pov, self.camera.axis)

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
        self.shaders["Colored"] = DefaultShader("Colored", self)
        self.shaders["Textured"] = TexturedShader("Textured", self)
        self.shaders["ColoredBump"] = ColoredBumpShader("ColoredBump", self)
        self.shaders["TexturedBump"] = TexturedBumpShader("TexturedBump", self)
        os.chdir(oldDir)

    def initScene(self, objects):
        keys = []
        [keys.append(item) for item in map(lambda mesh: mesh.appearance().material, objects) if item not in keys]

        groups = map(lambda key: filter(lambda mesh: mesh.appearance().material == key, objects), keys)
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
        self.updateModelView(self.camera.camera, self.camera.pov, self.camera.axis)
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
        self.updateModelView(self.camera.camera, self.camera.pov, self.camera.axis)
        glutPostRedisplay()

    def keyHandler(self, key, x, y):
        if key in ("\x1b", "q", "Q"):
            exit()
        elif key in ("1"):
            self.camera.side()
        elif key in ("3"):
            self.camera.front()
        elif key in ("7"):
            self.camera.top()
        elif key in ("."):
            self.camera.home()
        elif key in ("z", "Z"):
            self.wireframe = not self.wireframe
        self.updateModelView(self.camera.camera, self.camera.pov, self.camera.axis)
        glutPostRedisplay()

#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# rasterizer.py
# Copyright (C) 2013 xent
# Project is distributed under the terms of the GNU General Public License v3.0

import argparse
import os
import re
import subprocess

import Image
import ImageEnhance

from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GL.shaders import *
from OpenGL.GL.framebufferobjects import *

dpi = (90, 90) #Source dpi


def createShader(vertSource, fragSource):
    try:
        program = compileProgram(compileShader(vertSource, GL_VERTEX_SHADER), 
                                 compileShader(fragSource, GL_FRAGMENT_SHADER))
    except RuntimeError as runError:
        print runError.args[0] #Print error log
        print "Shader compilation failed"
        exit()
    except:
        print "Unknown shader error"
        exit()
    return program


class Render:
    def __init__(self):
        self.shaders = []
        #Framebuffer elements
        self.colorAttach = None
        self.colorOutput = None

        glutInit(sys.argv)
        glutInitDisplayMode(GLUT_RGBA | GLUT_SINGLE)
        glutInitWindowSize(32, 32)
        glutCreateWindow("Viewer")
        glutHideWindow()
        self.initScene()

    @staticmethod
    def readShader(name):
        fd = open("./shaders/%s.vert" % name, "rb")
        vertShader = fd.read()
        fd.close()
        fd = open("./shaders/%s.frag" % name, "rb")
        fragShader = fd.read()
        fd.close()
        return createShader(vertShader, fragShader)

    def loadShaders(self):
        oldDir = os.getcwd()
        scriptDir = os.path.dirname(os.path.realpath(__file__))
        if len(scriptDir) > 0:
            os.chdir(scriptDir)
        self.shaders = {}
        self.shaders["diffuse"] = Render.readShader("diffuse");
        self.shaders["normalmap"] = Render.readShader("normalmap");
        #glBindAttribLocation(self.shaders['normals'], 1, "tangent")
        #glLinkProgram(self.shaders['normals'])
        os.chdir(oldDir)

    def initScene(self):
        glClearColor(0.0, 0.0, 0.0, 0.0)
        self.loadShaders()

    def drawPlane(self):
        glBegin(GL_QUADS)
        glTexCoord2f(1.0, 1.0)
        glVertex3i(-1, -1, 0)
        glTexCoord2f(1.0, 0.0)
        glVertex3i(1, -1, 0)
        glTexCoord2f(0, 0)
        glVertex3i(1, 1, 0)
        glTexCoord2f(0, 1.0)
        glVertex3i(-1, 1, 0)
        glEnd()

    def createFramebuffer(self, size):
        if self.colorAttach is not None:
            glDeleteTextures([self.colorOutput])
        self.colorAttach = glGenFramebuffers(1)
        glBindFramebuffer(GL_FRAMEBUFFER, self.colorAttach)
        self.colorOutput = glGenTextures(1)
        glBindTexture(GL_TEXTURE_RECTANGLE, self.colorOutput)
        glTexImage2D(GL_TEXTURE_RECTANGLE, 0, GL_RGBA8, size[0], size[1], 0, GL_BGRA, GL_UNSIGNED_BYTE, None)
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_RECTANGLE, self.colorOutput, 0)
        glBindFramebuffer(GL_FRAMEBUFFER, 0)

    def processImage(self, size, textures, shader):
        layers = [GL_TEXTURE0, GL_TEXTURE1, GL_TEXTURE2, GL_TEXTURE3]
        texList = []
        position = 0
        for item in textures:
            data = item.tostring("raw", "RGBA", 0, -1)
            texID = glGenTextures(1)
            glBindTexture(GL_TEXTURE_RECTANGLE, texID)
            glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
            glTexImage2D(GL_TEXTURE_RECTANGLE, 0, GL_RGBA8, size[0], size[1], 0, GL_RGBA, GL_UNSIGNED_BYTE, data)
            texList.append((position, texID))
            position += 1
        self.createFramebuffer(size)

        glBindFramebuffer(GL_FRAMEBUFFER, self.colorAttach)
        buffers = [GL_COLOR_ATTACHMENT0]
        glDrawBuffers(1, buffers)
        glClear(GL_COLOR_BUFFER_BIT)
        glUseProgram(self.shaders[shader])

        for item in texList:
            glActiveTexture(layers[item[0]])
            glEnable(GL_TEXTURE_RECTANGLE)
            glTexParameterf(GL_TEXTURE_RECTANGLE, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            #glTexParameterf(GL_TEXTURE_RECTANGLE, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
            glTexParameterf(GL_TEXTURE_RECTANGLE, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
            glBindTexture(GL_TEXTURE_RECTANGLE, item[1])

        tex = glGetUniformLocation(self.shaders[shader], "silk") #TODO Fix order
        glUniform1i(tex, 0)
        tex = glGetUniformLocation(self.shaders[shader], "copper")
        glUniform1i(tex, 1)
        tex = glGetUniformLocation(self.shaders[shader], "mask")
        glUniform1i(tex, 2)
        color = glGetUniformLocation(self.shaders[shader], "padColor")
        glUniform3f(color, 1.0, 0.890, 0.0) #FIXME
        color = glGetUniformLocation(self.shaders[shader], "maskColor")
        glUniform3f(color, 0.039, 0.138, 0.332) #FIXME
        color = glGetUniformLocation(self.shaders[shader], "silkColor")
        glUniform3f(color, 1.0, 1.0, 1.0) #FIXME
        #dim = glGetUniformLocation(self.shaders['diffuse'], "dimensions") #FIXME Remove?
        #glUniform2i(dim, size[0], size[1])

        glViewport(0, 0, size[0], size[1])
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        self.drawPlane()

        pix = glReadPixels(0, 0, size[0], size[1], GL_RGBA, GL_UNSIGNED_BYTE)
        return Image.fromstring("RGBA", size, pix, "raw", "RGBA", 0, -1)


parser = argparse.ArgumentParser()
parser.add_argument("-i", dest="source", help="Source SVG image.", default="")
parser.add_argument("-o", dest="output", help="Output directory.", default="")
options = parser.parse_args()

baseName = re.search("(.*?)-(F|B)(ront|ack)\.svg", options.source, re.S)
layerSilk, layerCu, layerMask, outName = "", "", "", ""
if baseName:
    layerCu = "%s-%s%s" % (baseName.group(1), baseName.group(2), baseName.group(3))
    layerSilk = "%s-%s_SilkS" % (baseName.group(1), baseName.group(2))
    layerMask = "%s-%s_Mask" % (baseName.group(1), baseName.group(2))
    outName = "%s-%s_Render" % (baseName.group(1), baseName.group(2))

#TODO Replace with something command-line
for entry in (layerSilk, layerCu, layerMask):
    if not os.path.isfile("%s.svg" % entry):
        print "File does not exist: %s.svg" % entry
        exit()
    #subprocess.call(["inkscape", "%s.svg" % entry, "--verb", "FitCanvasToDrawing", "--verb", "FileSave", "--verb", \
            #"FileClose"])
    #subprocess.call(["inkscape", "-f", "%s.svg" % entry, "--export-dpi", "900", "-e", "%s.png" % entry])
    #stdout=subprocess.PIPE
    #subprocess.call(["rsvg", "--x-zoom=10.0", "--y-zoom=10.0", "--format=png", "%s.svg" % entry, "%s.png" % entry])

images = {}
imageFiles = {"copper": layerSilk, "silk": layerCu, "mask": layerMask}
for imgType in ("copper", "silk", "mask"):
    im = Image.open("%s.png" % imageFiles[imgType])
    im.load()
    images[imgType] = im
width, height = images["copper"].size[0], images["copper"].size[1]

rend = Render()
#im = rend.processImage((width, height), [images["copper"], images["silk"], images["mask"]], "diffuse")
#im.save("%s.png" % outName, 'PNG', dpi=(dpi[0] * 5.0, dpi[1] * 5.0))
im = rend.processImage((width, height), [images["copper"], images["silk"], images["mask"]], "normalmap")
im.save("%s_map.png" % outName, 'PNG', dpi=(dpi[0] * 5.0, dpi[1] * 5.0))

print "Image size: %dx%d" % (width, height)

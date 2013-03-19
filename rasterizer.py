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

#TODO
dpi = (90, 90) #Destination dpi


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
        fd = open("./shaders_proc/%s.vert" % name, "rb")
        vertShader = fd.read()
        fd.close()
        fd = open("./shaders_proc/%s.frag" % name, "rb")
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

    #TODO Split into two different functions for diffuse texture and normal map
    def processImage(self, size, textures, shader, colors, mirror = False):
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

        for item in texList:
            glActiveTexture(layers[item[0]])
            glEnable(GL_TEXTURE_RECTANGLE)
            glTexParameterf(GL_TEXTURE_RECTANGLE, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            glTexParameterf(GL_TEXTURE_RECTANGLE, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
            glBindTexture(GL_TEXTURE_RECTANGLE, item[1])

        glUseProgram(self.shaders[shader])
        tex = glGetUniformLocation(self.shaders[shader], "copper")
        glUniform1i(tex, 0)
        tex = glGetUniformLocation(self.shaders[shader], "silk")
        glUniform1i(tex, 1)
        tex = glGetUniformLocation(self.shaders[shader], "mask")
        glUniform1i(tex, 2)
        color = glGetUniformLocation(self.shaders[shader], "padColor")
        glUniform3f(color, colors["plating"][0], colors["plating"][1], colors["plating"][2])
        color = glGetUniformLocation(self.shaders[shader], "maskColor")
        glUniform3f(color, colors["mask"][0], colors["mask"][1], colors["mask"][2])
        color = glGetUniformLocation(self.shaders[shader], "silkColor")
        glUniform3f(color, colors["silk"][0], colors["silk"][1], colors["silk"][2])

        orient = glGetUniformLocation(self.shaders[shader], "orientation")
        val = -1.0 if mirror else 1.0
        glUniform1f(orient, val)

        glViewport(0, 0, size[0], size[1])
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        self.drawPlane()
        glDisable(GL_TEXTURE_RECTANGLE)
        glUseProgram(0)

        pix = glReadPixels(0, 0, size[0], size[1], GL_RGBA, GL_UNSIGNED_BYTE)
        return Image.fromstring("RGBA", size, pix, "raw", "RGBA", 0, -1)


#FIXME Move to separate module
def calcBorders(path):
    stream = open(path, "rb")
    edgeCuts = stream.read()
    stream.close()
    lineSearch = re.compile("<path.+?d=\"(.*?)\".*?>", re.S)
    pos = 0
    position = [None, None]
    while 1:
        content = lineSearch.search(edgeCuts, pos)
        if content is None:
            break
        data = []
        unformatted = content.group(1).replace("M", "").replace("L", "").splitlines()
        for part in unformatted:
            data.extend(part.split(' '))
        coords = ((float(data[0]), float(data[1])), (float(data[2]), float(data[3])))
        for point in coords:
            if position[0] is None:
                position[0] = [point[0], point[1]]
            else:
                if position[0][0] > point[0]:
                    position[0][0] = point[0]
                if position[0][1] > point[1]:
                    position[0][1] = point[1]
            if position[1] is None:
                position[1] = [point[0], point[1]]
            else:
                if position[1][0] < point[0]:
                    position[1][0] = point[0]
                if position[1][1] < point[1]:
                    position[1][1] = point[1]
        pos = content.end()
    return position

parser = argparse.ArgumentParser()
parser.add_argument("-i", dest="path", help="project directory", default="")
parser.add_argument("-p", dest="project", help="project name", default="")
parser.add_argument("-o", dest="output", help="output directory", default="")
parser.add_argument("--mask", dest="mask", help="mask color", default="10,35,85")
parser.add_argument("--silk", dest="silk", help="silk color", default="255,255,255")
parser.add_argument("--plating", dest="plating", help="plating color", default="255,228,0")
options = parser.parse_args()

if options.path[-1] != '/':
    options.path += '/'
if options.project == "":
    options.project = os.path.basename(options.path[0:-1])
    if options.project == "":
        print "Project name is empty"
        exit()
    print "Detected project name: %s" % options.project
if options.output == "":
    outPath = options.path
else:
    outPath = options.output

boardPos = [[0, 0], [0, 0]]
edgeFile = "%s%s-Edge_Cuts.svg" % (options.path, options.project)
if os.path.isfile(edgeFile):
    boardPos = calcBorders(edgeFile)
    for i in range(0, len(boardPos)):
        for j in range(0, len(boardPos[i])):
            boardPos[i][j] = boardPos[i][j] / 10000
        boardPos[i][1] = 8.26799 - boardPos[i][1] #FIXME Remove hardcoded values
else:
    print "%s not found" % edgeFile
    exit()

colors = {"mask": (), "silk": (), "plating": ()}
for color in [("mask", options.mask), ("silk", options.silk), ("plating", options.plating)]:
    splitted = color[1].split(",")
    #TODO Add value checking
    try:
        colors[color[0]] = (float(splitted[0]) / 256, float(splitted[1]) / 256, float(splitted[2]) / 256)
    except ValueError:
        print "Wrong color parameter: %s" % color[1]

layerList = []
for layer in [("Front", "F"), ("Back", "B")]:
    layerFile = "%s%s-%s.svg" % (options.path, options.project, layer[0])
    if os.path.isfile(layerFile):
        layerCu = "%s-%s" % (options.project, layer[0])
        layerSilk = "%s-%s_SilkS" % (options.project, layer[1])
        layerMask = "%s-%s_Mask" % (options.project, layer[1])
        layerDiffuse = "%s-%s_Diffuse" % (options.project, layer[0]) #Diffuse texture
        layerNormal = "%s-%s_Normals" % (options.project, layer[0]) #Normal map
        #TODO Improve error handling
        try:
            for entry in (layerCu, layerSilk, layerMask):
                filePath = "%s%s.svg" % (options.path, layerCu)
                if not os.path.isfile(filePath):
                    print "Not found: %s" % filePath
                    raise Exception()
        except:
            continue
        mirror = True if layer[0] == "Front" else False
        layerList.append(((layerCu, layerSilk, layerMask), {"diffuse": layerDiffuse, "normals": layerNormal}, mirror))
    else:
        print "Layer file does not exist: %s" % layerFile

rend = Render()
for layer in layerList:
    images = []
    for entry in layer[0]:
        convert = False
        if os.path.isfile("%s%s.png" % (outPath, entry)):
            srcTime = os.path.getctime("%s%s.svg" % (options.path, entry))
            dstTime = os.path.getctime("%s%s.png" % (outPath, entry))
            if srcTime > dstTime:
                convert = True
        else:
            convert = True
        if convert:
            subprocess.call(["inkscape", "-f", "%s%s.svg" % (options.path, entry), "--export-dpi", "900", "-e", \
                    "%s%s.png" % (outPath, entry), "-a", "%u:%u:%u:%u" % \
                    (int(boardPos[0][0] * 90), int(boardPos[0][1] * 90), \
                    int(boardPos[1][0] * 90), int(boardPos[1][1] * 90))])
        tmp = Image.open("%s%s.png" % (outPath, entry))
        tmp.load()
        images.append(tmp)
    width, height = images[0].size
    #TODO dpi=(dpi[0] * 5.0, dpi[1] * 5.0)
    #Diffuse texture
    processed = rend.processImage((width, height), [images[0], images[1], images[2]], "diffuse", colors, layer[2])
    processed.save("%s%s.png" % (outPath, layer[1]["diffuse"]), "PNG")
    #Normal map
    processed = rend.processImage((width, height), [images[0], images[1], images[2]], "normalmap", colors, layer[2])
    processed.save("%s%s.png" % (outPath, layer[1]["normals"]), "PNG")
    print "Image size: %dx%d" % (width, height)

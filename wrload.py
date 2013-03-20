#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# wrload.py
# Copyright (C) 2011 xent
# Project is distributed under the terms of the GNU General Public License v3.0

import re
import math
import numpy
import sys
import time
import argparse
import os
import subprocess
import Image

debug = False
opengl = True
try:
    from OpenGL.GL import *
    from OpenGL.GLU import *
    from OpenGL.GLUT import *
    from OpenGL.GL.shaders import *
except:
    opengl = False

def pDebug(text, level = 1):
    if debug:
        print text

def getAngle(v1, v2):
    mag1 = math.sqrt(v1[0]*v1[0] + v1[1]*v1[1] + v1[2]*v1[2])
    mag2 = math.sqrt(v2[0]*v2[0] + v2[1]*v2[1] + v2[2]*v2[2])
    res = (v1[0]*v2[0] + v1[1]*v2[1] + v1[2]*v2[2]) / (mag1 * mag2)
    ac = math.acos(res)
    if v2[0]*v1[1] - v2[1]*v1[0] < 0:
        ac *= -1
    return ac

def normalize(vect):
    val = numpy.linalg.norm(vect)
    if val != 0:
        return vect / val
    else:
        return vect

def getNormal(v1, v2):
    return numpy.matrix([[float(v1[1] * v2[2] - v1[2] * v2[1])],
                         [float(v1[2] * v2[0] - v1[0] * v2[2])],
                         [float(v1[0] * v2[1] - v1[1] * v2[0])]])

def getTangent(v1, v2, st1, st2):
    div = st1[1] * st2[0] - st1[0] * st2[1]
    if div == 0:
        return numpy.array([0.0, 0.0, 1.0])
    coef = 1. / div
    return numpy.array([coef * (v1[0] * -st2[1] + v2[0] * st1[1]),
                        coef * (v1[1] * -st2[1] + v2[1] * st1[1]),
                        coef * (v1[2] * -st2[1] + v2[2] * st1[1])])

def fillRotateMatrix(v, angle):
    cs = math.cos(angle)
    sn = math.sin(angle)
    v = [float(v[0]), float(v[1]), float(v[2])]
    return numpy.matrix([[     cs + v[0]*v[0]*(1 - cs), v[0]*v[1]*(1 - cs) - v[2]*sn, v[0]*v[2]*(1 - cs) + v[1]*sn, 0.],
                         [v[1]*v[0]*(1 - cs) + v[2]*sn,      cs + v[1]*v[1]*(1 - cs), v[1]*v[2]*(1 - cs) - v[0]*sn, 0.],
                         [v[2]*v[0]*(1 - cs) - v[1]*sn, v[2]*v[1]*(1 - cs) + v[0]*sn,      cs + v[2]*v[2]*(1 - cs), 0.],
                         [                          0.,                           0.,                           0., 1.]])

def skipChunk(fd):
    balance = 1
    while 1:
        data = fd.readline()
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
    balance, offset, update = 0, 0, False
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

def extractMat(obj):
    if isinstance(obj, vrmlMaterial):
        return obj
    for entry in obj.objects:
        res = extractMat(entry)
        if res is not None:
            return res
    return None


class vrmlEntry:
    identifier = 0
    def __init__(self, parent=None):
        self.id = None
        self.parent = parent
        self.name = ""
        self.objects = []
        self.active = True #Using in rendering and rebuilding
        if self.parent:
            self._level = self.parent._level + 2
        else:
            self._level = 0

    def read(self, fd, objFilter=""):
        defPattern = re.compile("([\w]*?)\s*([\w\.\-]*?)\s*(\w+)\s*{", re.I | re.S)
        delta, offset, balance = 0, 0, 0
        if len(objFilter) != 0:
            namePattern = re.compile(objFilter, re.I | re.S)
        else:
            namePattern = None
        #Highest level
        while 1:
            data = fd.readline()
            if len(data) == 0:
                break
            #print "%s - %d: '%s'" % (self.__class__.__name__, balance, data.replace("\n", "").replace("\r", ""))
            regexp = defPattern.search(data)
            if regexp:
                (delta, offset) = calcBalance(data[:regexp.start()], -1, ('{'), ('}'))
                balance += delta
                initialPos = fd.tell()
                self.readSpecific(fd, data[:regexp.start()])
                if initialPos != fd.tell():
                    print "%sRead error" % (' ' * self._level)
                    break
                if balance < 0:
                    pDebug("%sWrong balance: %d" % (' ' * self._level, balance))
                    fd.seek(-(len(data) - regexp.start() + offset), os.SEEK_CUR)
                    break
                fd.seek(-(len(data) - regexp.end()), os.SEEK_CUR)
                entry = None
                pDebug("%sEntry: '%s' '%s' '%s' Balance: %d" % (' ' * self._level, regexp.group(1), regexp.group(2), regexp.group(3), balance))
                entryType = regexp.group(3)
                try:
                    if isinstance(self, vrmlScene) or isinstance(self, vrmlTransform) or isinstance(self, vrmlInline):
                        #Collision and Switch nodes functionality unimplemented
                        if entryType in ("Transform", "Group", "Collision", "Switch"):
                            entry = vrmlTransform(self)
                            #Transform filtering
                            if namePattern and len(regexp.group(2)) != 0:
                                res = namePattern.search(regexp.group(2))
                                if not res:
                                    pDebug("%sTransform name does not match with pattern" % (' ' * self._level))
                                    entry.active = False
                        elif entryType == "Inline":
                            entry = vrmlInline(self)
                        elif entryType == "Shape":
                            entry = vrmlShape(self)
                        else:
                            raise Exception()
                    elif isinstance(self, vrmlShape):
                        if entryType == "Appearance":
                            entry = vrmlAppearance(self)
                        elif entryType == "IndexedFaceSet":
                            entry = vrmlGeometry(self)
                        else:
                            raise Exception()
                    elif isinstance(self, vrmlAppearance):
                        if entryType == "Material":
                            entry = vrmlMaterial(self)
                        elif entryType == "ImageTexture":
                            entry = vrmlTexture(self)
                        else:
                            raise Exception()
                    elif isinstance(self, vrmlGeometry):
                        if entryType == "Coordinate":
                            entry = vrmlCoords(self, 'model')
                        elif entryType == "TextureCoordinate":
                            entry = vrmlCoords(self, 'texture')
                        else:
                            raise Exception()
                    else:
                        raise Exception()
                except:
                    pDebug("%sUnsopported chunk sequence: %s > %s" % (' ' * self._level, self.__class__.__name__, entryType))
                    offset = skipChunk(fd)
                    fd.seek(-offset, os.SEEK_CUR)
                if entry:
                    if regexp.group(1) == "DEF" and len(regexp.group(2)) > 0:
                        entry.name = regexp.group(2)
                    entry.read(fd, objFilter)
                    ptr = self
                    inline = None
                    while not isinstance(ptr, vrmlScene):
                        if inline is None and isinstance(ptr, vrmlInline):
                            inline = ptr
                        ptr = ptr.parent
                    duplicate = False
                    for current in ptr.entries: #Search for duplicates
                        if entry == current:
                            pDebug("%sNot unique, using entry with id: %d" % (' ' * self._level, current.id))
                            entry = current
                            duplicate = True
                            break
                    self.objects.append(entry)
                    if inline:
                        inline.entries.append(entry)
                    if not duplicate:
                        entry.id = vrmlEntry.identifier
                        vrmlEntry.identifier += 1
                        ptr.entries.append(entry)
            else:
                (delta, offset) = calcBalance(data, -(balance + 1), ('{'), ('}'))
                balance += delta
                initialPos = fd.tell()
                self.readSpecific(fd, data)
                using = re.search("USE\s+([\w\.\-]+)", data, re.I | re.S)
                if using and using.start() < len(data) - offset:
                    pDebug("%sUsing entry %s" % (' ' * self._level, using.group(1)))
                    ptr = self
                    while not isinstance(ptr, vrmlScene) and not isinstance(ptr, vrmlInline):
                        ptr = ptr.parent
                    for obj in ptr.entries:
                        if obj.name == using.group(1):
                            pDebug("%sFound entry %s" % (' ' * self._level, using.group(1)))
                            self.objects.append(obj)
                if balance < 0:
                    pDebug("%sBalance mismatch: %d" % (' ' * self._level, balance))
                    if initialPos == fd.tell():
                        fd.seek(-offset, os.SEEK_CUR)
                    break

    def readSpecific(self, fd, string):
        pass


class vrmlScene(vrmlEntry):
    def __init__(self):
        vrmlEntry.__init__(self)
        self.entries = []
        self.transform = numpy.matrix([[1., 0., 0., 0.],
                                       [0., 1., 0., 0.],
                                       [0., 0., 1., 0.],
                                       [0., 0., 0., 1.]])

    def setTransform(self, aTrans, aRot, aScale):
        #translation = numpy.matrix([[1., 0., 0., aTrans[0]],
                                    #[0., 1., 0., aTrans[1]],
                                    #[0., 0., 1., aTrans[2]],
                                    #[0., 0., 0., 1.]])
        translation = numpy.matrix([[0., 0., 0., aTrans[0]],
                                    [0., 0., 0., aTrans[1]],
                                    [0., 0., 0., aTrans[2]],
                                    [0., 0., 0., 0.]])
        rotation = fillRotateMatrix(aRot, aRot[3]);
        scale = numpy.matrix([[aScale[0], 0., 0., 0.],
                              [0., aScale[1], 0., 0.],
                              [0., 0., aScale[2], 0.],
                              [0., 0., 0., 1.]])
        #self.transform = translation * rotation * scale
        self.transform = translation + rotation * scale

    def loadFile(self, fileName, objFilter=""):
        wrlFile = open(fileName, "rb")
        oldDir = os.getcwd()
        if len(os.path.dirname(fileName)) > 0:
            os.chdir(os.path.dirname(fileName))
        self.read(wrlFile, objFilter)
        os.chdir(oldDir)
        wrlFile.close()

        #Transparency ordering
        latest = None
        for i in range(len(self.objects) - 1, -1, -1):
            mat = extractMat(self.objects[i])
            if mat is not None and mat.transparency > 0.0:
                if not latest:
                    continue
                else:
                    self.objects[i], self.objects[latest] = self.objects[latest], self.objects[i]
                    latest = latest - 1
            else:
                if not latest:
                    latest = i

    def saveFile(self, fileName):
        wrlFile = open(fileName, "wb")
        compList = []
        wrlFile.write("#VRML V2.0 utf8\n#Exported from Blender by wrlconv.py\n")
        for entry in self.objects:
            if entry.active:
                entry.write(wrlFile, compList, self.transform, entry.active)
        wrlFile.close()

    def buildMesh(self):
        class groupDescriptor:
            def __init__(self, appearance):
                self.appearance = appearance
                self.count = numpy.array([0, 0])
                self.name = ""
                self.objects = []

        def groupObjects(top, grps, granted=False):
            for entry in top.objects:
                if not entry.active and not granted:
                    continue
                if isinstance(top, vrmlScene) or isinstance(top, vrmlTransform):
                    groupObjects(entry, grps, entry.active or granted)
                if isinstance(entry, vrmlShape):
                    count = numpy.array([0, 0])
                    for geo in entry.objects:
                        if isinstance(geo, vrmlGeometry):
                            count += numpy.array([geo.triCount, geo.quadCount])
                    for app in entry.objects:
                        if isinstance(app, vrmlAppearance):
                            if app.id not in grps.keys():
                                grps[app.id] = groupDescriptor(app)
                                for appName in app.objects:
                                    if len(grps[app.id].name) != 0:
                                        grps[app.id].name += ", "
                                    grps[app.id].name += appName.name
                            grps[app.id].count += count
                            grps[app.id].objects.append(entry)
                            break

        def buildObjects(top, objlist, mesh, transform, off, app, granted=False): #FIXME rewrite
            for entry in top.objects:
                if not entry.active and not granted:
                    continue
                if isinstance(entry, vrmlScene) or isinstance(entry, vrmlTransform):
                    off = buildObjects(entry, objlist, mesh, transform * entry.transform, off, app, entry.active or granted)
                if entry in objlist:
                    off = entry.mesh(meshobj, off, app, transform)
            return off

        groups = {}
        groupObjects(self, groups)
        pDebug("Objects grouped, total groups: %d" % len(groups))
        res = []
        for i in groups.keys(): #FIXME rewrite
            pDebug("Group with appearance id: %d (%s), object count: %d" % (groups[i].appearance.id, groups[i].name, len(groups[i].objects)))
            meshobj = mesh()
            length = groups[i].count[0] + groups[i].count[1]
            meshobj.vertexList = numpy.zeros(length * 3, dtype = numpy.float32)
            meshobj.normalList = numpy.zeros(length * 3, dtype = numpy.float32)
            meshobj.tangentList = numpy.zeros(length * 3, dtype = numpy.float32) #FIXME
            meshobj.texList = numpy.zeros(length * 2, dtype = numpy.float32) #FIXME
            offsets = (0, groups[i].count[0]) #(Triangles, Quads)
            offsets = buildObjects(self, groups[i].objects, meshobj, self.transform, offsets, groups[i].appearance)
            fs = faceset()
            if offsets[0] > 0:
                fs.append(GL_TRIANGLES, 0, offsets[0])
            if offsets[1] > offsets[0]:
                fs.append(GL_QUADS, offsets[0], offsets[1] - offsets[0])
            meshobj.appearance = groups[i].appearance
            meshobj.objects.append(fs)
            res.append(meshobj)

        #Transparency ordering
        latest = None
        for i in range(len(res) - 1, -1, -1):
            mat = extractMat(res[i].appearance)
            if mat is not None and mat.transparency > 0.0:
                res[i].zbuffer = False
                if not latest:
                    continue
                else:
                    res[i], res[latest] = res[latest], res[i]
                    latest = latest - 1
            else:
                if not latest:
                    latest = i
        return res


class vrmlTransform(vrmlEntry):
    def __init__(self, parent):
        vrmlEntry.__init__(self, parent)
        self.transform = numpy.matrix([[1., 0., 0., 0.],
                                       [0., 1., 0., 0.],
                                       [0., 0., 1., 0.],
                                       [0., 0., 0., 1.]])

    def readSpecific(self, fd, string):
        tmp = re.search("translation\s+([+e\d\.\-]+)\s+([+e\d\.\-]+)\s+([+e\d\.\-]+)", string, re.I | re.S)
        if tmp:
            #tform = numpy.matrix([[1., 0., 0., float(tmp.group(1))],
                                  #[0., 1., 0., float(tmp.group(2))],
                                  #[0., 0., 1., float(tmp.group(3))],
                                  #[0., 0., 0., 1.]])
            #self.transform = self.transform * tform
            tform = numpy.matrix([[0., 0., 0., float(tmp.group(1))],
                                  [0., 0., 0., float(tmp.group(2))],
                                  [0., 0., 0., float(tmp.group(3))],
                                  [0., 0., 0., 0.]])
            self.transform = self.transform + tform
        tmp = re.search("rotation\s+([+e\d\.\-]+)\s+([+e\d\.\-]+)\s+([+e\d\.\-]+)\s+([+e\d\.\-]+)", string, re.I | re.S)
        if tmp:
            tform = fillRotateMatrix([float(tmp.group(1)), float(tmp.group(2)), float(tmp.group(3))], float(tmp.group(4)))
            self.transform = self.transform * tform
        tmp = re.search("scale\s+([+e\d\.\-]+)\s+([+e\d\.\-]+)\s+([+e\d\.\-]+)", string, re.I | re.S)
        if tmp:
            tform = numpy.matrix([[float(tmp.group(1)), 0., 0., 0.],
                                  [0., float(tmp.group(2)), 0., 0.],
                                  [0., 0., float(tmp.group(3)), 0.],
                                  [0., 0., 0., 1.]])
            self.transform = self.transform * tform

    def write(self, fd, compList, transform, granted=False):
        tform = transform * self.transform
        for obj in self.objects:
            if obj.active or granted:
                obj.write(fd, compList, tform, obj.active or granted)

class vrmlInline(vrmlTransform):
    def __init__(self, parent):
        vrmlTransform.__init__(self, parent)
        self.entries = []

    def readSpecific(self, fd, string):
        urlSearch = re.search("url\s+\"([\w\-\._\/]+)\"", string, re.S)
        if urlSearch:
            oldDir = os.getcwd()
            if os.path.isfile(urlSearch.group(1)):
                #print "%sLoading file: %s" % (' ' * self._level, urlSearch.group(1))
                #self.name = urlSearch.group(1)
                wrlFile = open(urlSearch.group(1), "r")
                if len(os.path.dirname(urlSearch.group(1))) > 0:
                    os.chdir(os.path.dirname(urlSearch.group(1)))
                self.read(wrlFile)
                wrlFile.close()
            else:
                print "Inline file not found: %s" % urlSearch.group(1)
            os.chdir(oldDir)


class vrmlShape(vrmlEntry):
    _vcount = 0
    _pcount = 0
    def __init__(self, parent):
        vrmlEntry.__init__(self, parent)

    def mesh(self, meshObject, offsets, appearance, transform):
        #print "%sDraw shape %s" % (' ' * 2, self.name)
        (triOffset, quadOffset) = (offsets[0], offsets[1])
        _tsa = time.time()
        for obj in self.objects:
            if isinstance(obj, vrmlGeometry):
                #TODO add solid parsing
                genTex = obj.polygonsUV != None and len(obj.polygons) == len(obj.polygonsUV)
                vertices = []
                verticesUV = []
                for coords in obj.objects:
                    if isinstance(coords, vrmlCoords) and coords.cType == vrmlCoords.TYPE["model"]:
                        for vert in coords.vertices:
                            tmp = numpy.matrix([[vert[0]], [vert[1]], [vert[2]], [1.]])
                            tmp = transform * tmp
                            vertices.append(numpy.array([float(tmp[0]), float(tmp[1]), float(tmp[2])]))
                    elif isinstance(coords, vrmlCoords) and coords.cType == vrmlCoords.TYPE["texture"]:
                        for vert in coords.vertices:
                            verticesUV.append(vert)
                if not obj.smooth: #Flat shading
                    for poly in range(0, len(obj.polygons)):
                        if genTex:
                            tangent = getTangent(vertices[obj.polygons[poly][1]] - vertices[obj.polygons[poly][0]],
                                                 vertices[obj.polygons[poly][2]] - vertices[obj.polygons[poly][0]],
                                                 verticesUV[obj.polygonsUV[poly][1]] - verticesUV[obj.polygonsUV[poly][0]],
                                                 verticesUV[obj.polygonsUV[poly][2]] - verticesUV[obj.polygonsUV[poly][0]])
                            tangent = normalize(tangent)
                        normal = getNormal(vertices[obj.polygons[poly][1]] - vertices[obj.polygons[poly][0]],
                                           vertices[obj.polygons[poly][2]] - vertices[obj.polygons[poly][0]])
                        normal = normalize(normal)
                        normal = numpy.array([float(normal[0]), float(normal[1]), float(normal[2])])
                        if len(obj.polygons[poly]) == 3:
                            pos = triOffset
                            triOffset += 3
                        else:
                            pos = quadOffset
                            quadOffset += 4
                        for ind in range(0, len(obj.polygons[poly])):
                            meshObject.vertexList[3 * pos:3 * pos + 3] = vertices[obj.polygons[poly][ind]][0:3]
                            meshObject.normalList[3 * pos:3 * pos + 3] = normal
                            if genTex:
                                meshObject.texList[2 * pos:2 * pos + 2] = verticesUV[obj.polygonsUV[poly][ind]][0:2]
                                meshObject.tangentList[3 * pos:3 * pos + 3] = tangent[0:3]
                            pos += 1
                else: #Smooth shading
                    normals = []
                    tangents = []
                    for i in range(0, len(vertices)):
                        normals.append(numpy.array([0., 0., 0.,]))
                    if genTex:
                        for i in range(0, len(vertices)):
                            tangents.append(numpy.array([0., 0., 0.,]))
                    for poly in range(0, len(obj.polygons)):
                        if genTex:
                            tangent = getTangent(vertices[obj.polygons[poly][1]] - vertices[obj.polygons[poly][0]],
                                                 vertices[obj.polygons[poly][2]] - vertices[obj.polygons[poly][0]],
                                                 verticesUV[obj.polygonsUV[poly][1]] - verticesUV[obj.polygonsUV[poly][0]],
                                                 verticesUV[obj.polygonsUV[poly][2]] - verticesUV[obj.polygonsUV[poly][0]])
                            tangent = normalize(tangent)
                            tangent = numpy.array([float(tangent[0]), float(tangent[1]), float(tangent[2])])
                        normal = getNormal(vertices[obj.polygons[poly][1]] - vertices[obj.polygons[poly][0]], 
                                           vertices[obj.polygons[poly][2]] - vertices[obj.polygons[poly][0]])
                        normal = normalize(normal)
                        for ind in obj.polygons[poly]:
                            normals[ind] += numpy.array([float(normal[0]), float(normal[1]), float(normal[2])])
                            if genTex:
                                tangents[ind] += tangent
                    for i in range(0, len(vertices)):
                        normals[i] = normalize(normals[i])
                        if genTex:
                            tangents[i] = normalize(tangents[i])
                    for poly in range(0, len(obj.polygons)):
                        if len(obj.polygons[poly]) == 3:
                            pos = triOffset
                            triOffset += 3
                        else:
                            pos = quadOffset
                            quadOffset += 4
                        for ind in range(0, len(obj.polygons[poly])):
                            meshObject.vertexList[3 * pos:3 * pos + 3] = vertices[obj.polygons[poly][ind]][0:3]
                            meshObject.normalList[3 * pos:3 * pos + 3] = normals[obj.polygons[poly][ind]][0:3]
                            if genTex:
                                meshObject.texList[2 * pos:2 * pos + 2] = verticesUV[obj.polygonsUV[poly][ind]][0:2]
                                meshObject.tangentList[3 * pos:3 * pos + 3] = tangents[obj.polygons[poly][ind]][0:3]
                            pos += 1
                _tsb = time.time()
                #Vertex count needed in VBO rendering
                #FIXME replace with vertex count used in current shape
                localVCount = (triOffset - offsets[0]) + (quadOffset - offsets[1])
                vrmlShape._vcount += localVCount
                vrmlShape._pcount += len(obj.polygons)
                pDebug("%sCreated in: %f, vertices: %d, polygons: %d" % (' ' * 2, _tsb - _tsa, localVCount, len(obj.polygons)))
        return (triOffset, quadOffset)

    def reindex(self):
        #Return vertex and polygon lists
        vertices = []
        polygons = []
        usedVertices = []
        translatedVertices = {}
        totalVertices = 0
        #FIXME optimize
        for obj in self.objects:
            if isinstance(obj, vrmlGeometry):
                for poly in obj.polygons:
                    for ind in poly:
                        if ind not in usedVertices:
                            usedVertices.append(ind)
        usedVertices.sort()
        for obj in self.objects:
            if isinstance(obj, vrmlGeometry):
                for coords in obj.objects:
                    if isinstance(coords, vrmlCoords) and coords.cType == vrmlCoords.TYPE["model"]:
                        totalVertices = len(coords.vertices)
                        for i in range(0, len(usedVertices)):
                            vertices.append(coords.vertices[usedVertices[i]])
                            translatedVertices[usedVertices[i]] = i
        for obj in self.objects:
            if isinstance(obj, vrmlGeometry):
                for poly in obj.polygons:
                    newPoly = []
                    for ind in poly:
                        newPoly.append(translatedVertices[ind])
                    polygons.append(newPoly)
        pDebug("  Shape reindexed: %d/%d vertices, %d polygons" % (len(vertices), totalVertices, len(polygons)))
        return (vertices, polygons)

    def write(self, fd, compList, transform, granted=False):
        partialGeo = False
        currentMat = extractMat(self)
        if self.parent and self.parent.name != "":
            if len(self.parent.objects) == 1:
                shapeName = self.parent.name
            else:
                shapeName = "%s_%d" % (self.parent.name, self.parent.objects.index(self))
                partialGeo = True
            pDebug("Write object %s" % shapeName)
            fd.write("DEF %s Transform {\n  children [\n" % shapeName)
        else:
            pDebug("Write untitled object")
            fd.write("Transform {\n  children [\n")
        fd.write("    Shape {\n")
        currentMat.write(fd, compList)
        if partialGeo:
            (vert, poly) = self.reindex()
            fd.write("      geometry IndexedFaceSet {\n        coord Coordinate { point [\n")
            for i in range(0, len(vert)):
                tmp = numpy.matrix([[vert[i][0]], [vert[i][1]], [vert[i][2]], [1.]])
                tmp = transform * tmp
                fd.write("          %f %f %f" % (float(tmp[0]), float(tmp[1]), float(tmp[2])))
                if i != len(vert) - 1:
                    fd.write(",\n")
            fd.write(" ] }\n")
            fd.write("        coordIndex [\n")
            for i in range(0, len(poly)):
                fd.write("          ")
                for ind in poly[i]:
                    fd.write("%d, " % ind)
                fd.write("-1")
                if i != len(poly) - 1:
                    fd.write(",\n")
            fd.write(" ]\n")
            fd.write("      }\n")
        else:
            for obj in self.objects:
                if isinstance(obj, vrmlGeometry):
                    fd.write("      geometry IndexedFaceSet {\n        coord Coordinate { point [\n")
                    for coords in obj.objects:
                        if isinstance(coords, vrmlCoords) and coords.cType == vrmlCoords.TYPE["model"]:
                            for i in range(0, len(coords.vertices)):
                                tmp = numpy.matrix([[coords.vertices[i][0]], [coords.vertices[i][1]], [coords.vertices[i][2]], [1.]])
                                tmp = transform * tmp
                                fd.write("          %f %f %f" % (float(tmp[0]), float(tmp[1]), float(tmp[2])))
                                if i != len(coords.vertices) - 1:
                                    fd.write(",\n")
                    fd.write(" ] }\n")
                    fd.write("        coordIndex [\n")
                    for i in range(0, len(obj.polygons)):
                        fd.write("          ")
                        for ind in obj.polygons[i]:
                            fd.write("%d, " % ind)
                        fd.write("-1")
                        if i != len(obj.polygons) - 1:
                            fd.write(",\n")
                    fd.write(" ]\n")
                    fd.write("      }\n")
        fd.write("    }\n")
        fd.write("  ]\n}\n\n")


class vrmlGeometry(vrmlEntry):
    def __init__(self, parent):
        vrmlEntry.__init__(self, parent)
        self.smooth     = False
        self.solid      = False
        self.polygons   = None
        self.triCount   = 0
        self.quadCount  = 0
        self.polygonsUV = None

    def readSpecific(self, fd, string):
        #print "%sTry geo read: %s" % (' ' * self._level, string.replace("\n", "").replace("\t", ""))
        initialPos = fd.tell()

        paramSearch = re.search("solid\s+(TRUE|FALSE)", string, re.S)
        if paramSearch and paramSearch.group(1) == "TRUE":
            self.solid = True
        paramSearch = re.search("smooth\s+(TRUE|FALSE)", string, re.S)
        if paramSearch and paramSearch.group(1) == "TRUE":
            self.smooth = True

        coordSearch = re.search("coordIndex\s*\[", string, re.S)
        texSearch = re.search("texCoordIndex\s*\[", string, re.S)
        if not coordSearch and not texSearch:
            return
        polyPattern = re.compile("([ ,\t\d]+)-1", re.I | re.S)
        indPattern = re.compile("[ ,\t]*(\d+)[ ,\t]*", re.I | re.S)
        polygons = []
        delta, offset, balance = 0, 0, 0
        data = string
        if coordSearch:
            pDebug("%sStart coordinate polygons read" % (' ' * self._level))
            pPos = coordSearch.end()
        else:
            pDebug("%sStart texture polygons read" % (' ' * self._level))
            pPos = texSearch.end()
        while 1:
            while 1:
                regexp = polyPattern.search(data, pPos)
                if regexp:
                    (delta, offset) = calcBalance(data[pPos:regexp.start()], -1, (), (']'))
                    balance += delta
                    offset = len(data) - regexp.start() + offset
                    if balance != 0:
                        pDebug("%sWrong balance: %d, offset: %d" % (' ' * self._level, balance, offset))
                        break
                    polyData = []
                    indPos = 0
                    while 1:
                        ind = indPattern.search(regexp.group(1), indPos)
                        if ind is None:
                            break
                        polyData.append(int(ind.group(1)))
                        indPos = ind.end()
                    if coordSearch:
                        if len(polyData) == 3:
                            self.triCount += 3
                            polygons.append(polyData)
                        elif len(polyData) == 4:
                            self.quadCount += 4
                            polygons.append(polyData)
                        else:
                            for tesselPos in range(1, len(polyData) - 1):
                                self.triCount += 3
                                polygons.append([polyData[0], polyData[tesselPos], polyData[tesselPos + 1]])
                    else:
                        polygons.append(polyData) #FIXME Optimize
                    pPos = regexp.end()
                else:
                    (delta, offset) = calcBalance(data, None, (), (']'))
                    balance += delta
                    offset = len(data)
                    break
            if balance != 0:
                if initialPos != fd.tell():
                    fd.seek(-offset, os.SEEK_CUR)
                pDebug("%sBalance mismatch: %d, offset: %d" % (' ' * self._level, balance, offset))
                break
            data = fd.readline()
            if len(data) == 0:
                break
            pPos = 0
        if coordSearch:
            self.polygons = polygons
            pDebug("%sRead poly done, %d tri, %d quad, %d vertices" %
                    (' ' * self._level, self.triCount / 3, self.quadCount / 4, self.triCount + self.quadCount))
        elif texSearch:
            self.polygonsUV = polygons
            pDebug("%sRead UV poly done, %d poly, %d vertices" % (' ' * self._level, len(self.polygonsUV), 0))

class vrmlCoords(vrmlEntry):
    TYPE = {'model': 0, 'texture': 1}
    def __init__(self, parent, cType):
        vrmlEntry.__init__(self, parent)
        self.cType = vrmlCoords.TYPE[cType]
        self.vertices = None

    def readSpecific(self, fd, string):
        initialPos = fd.tell()
        indexSearch = re.search("point\s*\[", string, re.S)
        if indexSearch:
            pDebug("%sStart vertex read, type: %s" % (' ' * self._level, vrmlCoords.TYPE.keys()[self.cType]))
            if self.cType == vrmlCoords.TYPE["model"]:
                vertexPattern = re.compile("([+e\d\-\.]+)[ ,\t]+([+e\d\-\.]+)[ ,\t]+([+e\d\-\.]+)", re.I | re.S)
            elif self.cType == vrmlCoords.TYPE["texture"]:
                vertexPattern = re.compile("([+e\d\-\.]+)[ ,\t]+([+e\d\-\.]+)", re.I | re.S)
            self.vertices = []
            delta, offset, balance = 0, 0, 0
            data = string
            vPos = indexSearch.end()
            while 1:
                while 1:
                    regexp = vertexPattern.search(data, vPos)
                    if regexp:
                        (delta, offset) = calcBalance(data[vPos:regexp.start()], -1, (), ('}'))
                        balance += delta
                        offset = len(data) - regexp.start() + offset
                        if initialPos != fd.tell():
                            offset += 1
                        if balance != 0:
                            pDebug("%sWrong balance: %d, offset: %d" % (' ' * self._level, balance, offset))
                            break
                        if self.cType == vrmlCoords.TYPE["model"]:
                            self.vertices.append(numpy.array([float(regexp.group(1)), float(regexp.group(2)), \
                                    float(regexp.group(3))]))
                        elif self.cType == vrmlCoords.TYPE["texture"]:
                            self.vertices.append(numpy.array([float(regexp.group(1)), float(regexp.group(2))]))
                        vPos = regexp.end()
                    else:
                        (delta, offset) = calcBalance(data[vPos:], -1, (), ('}'))
                        balance += delta
                        if initialPos != fd.tell():
                            offset += 1
                        break
                if balance != 0:
                    if initialPos != fd.tell():
                        fd.seek(-offset, os.SEEK_CUR)
                    pDebug("%sBalance mismatch: %d, offset: %d" % (' ' * self._level, balance, offset))
                    break
                data = fd.readline()
                if len(data) == 0:
                    break
                vPos = 0
            pDebug("%sEnd vertex read, count: %d" % (' ' * self._level, len(self.vertices)))


class vrmlAppearance(vrmlEntry):
    def __init__(self, parent=None):
        vrmlEntry.__init__(self, parent)
        self.diffuse = None
        self.normal = None

    def __eq__(self, other):
        if not isinstance(other, vrmlAppearance):
            return False
        if len(self.objects) != len(other.objects):
            return False
        for mat in self.objects:
            if mat not in other.objects:
                return False
        return True

    def __ne__(self, other):
        return not self == other


class vrmlMaterial(vrmlEntry):
    def __init__(self, parent=None):
        if not isinstance(parent, vrmlAppearance):
            raise Exception()
        vrmlEntry.__init__(self, parent)
        self.diffuseColor     = [1., 1., 1., 1.]
        self.ambientColor     = [1., 1., 1., 1.]
        self.specularColor    = [0., 0., 0., 1.]
        self.emissiveColor    = [0., 0., 0., 1.]
        self.ambientIntensity = 1.
        self.shininess        = 0.
        self.transparency     = 0.

    def __eq__(self, other):
        if not isinstance(other, vrmlMaterial):
            return False
        if self.diffuseColor  == other.diffuseColor and \
           self.ambientColor  == other.ambientColor and \
           self.specularColor == other.specularColor and \
           self.emissiveColor == other.emissiveColor and \
           self.shininess     == other.shininess:
            return True
        else:
            return False

    def __ne__(self, other):
        return not self == other

    def readSpecific(self, fd, string):
        #print "%sReading material: %s" % (' ' * self._level, string.replace("\n", "").replace("\t", ""))
        tmp = re.search("transparency\s+([+e\d\-\.]+)", string, re.I | re.S)
        if tmp:
            self.transparency = float(tmp.group(1))
        tmp = re.search("diffuseColor\s+([+e\d\-\.]+)\s+([+e\d\-\.]+)\s+([+e\d\-\.]+)", string, re.I | re.S)
        if tmp:
            self.diffuseColor = [float(tmp.group(1)), float(tmp.group(2)), float(tmp.group(3)), 1. - self.transparency]
        tmp = re.search("ambientIntensity\s+([+e\d\-\.]+)", string, re.I | re.S)
        if tmp:
            self.ambientIntensity = float(tmp.group(1))
        tmp = re.search("specularColor\s+([+e\d\-\.]+)\s+([+e\d\-\.]+)\s+([+e\d\-\.]+)", string, re.I | re.S)
        if tmp:
            self.specularColor = [float(tmp.group(1)), float(tmp.group(2)), float(tmp.group(3)), 1.]
        tmp = re.search("emissiveColor\s+([+e\d\-\.]+)\s+([+e\d\-\.]+)\s+([+e\d\-\.]+)", string, re.I | re.S)
        if tmp:
            self.emissiveColor = [float(tmp.group(1)), float(tmp.group(2)), float(tmp.group(3)), 1.]
        tmp = re.search("shininess\s+([+e\d\-\.]+)", string, re.I | re.S)
        if tmp:
            self.shininess = float(tmp.group(1))
        self.ambientColor = [self.diffuseColor[0] * self.ambientIntensity, 
                             self.diffuseColor[1] * self.ambientIntensity, 
                             self.diffuseColor[2] * self.ambientIntensity, 
                             1. - self.transparency]
        self.diffuseColor[3] = 1. - self.transparency

    def write(self, fd, compList):
        matName = self.name if self.name != "" else "Unnamed_%d" % self.id
        if not matName in compList:
            #ambInt = self.ambientIntensity * 3.
            #if ambInt > 1.:
                #ambInt = 1.
            #For KiCAD just set ambient intensity to 1.0
            ambInt = 1.
            fd.write("      appearance Appearance {\n        material DEF %s Material {\n" % matName)
            fd.write("          diffuseColor %f %f %f\n" % (self.diffuseColor[0], self.diffuseColor[1], self.diffuseColor[2]))
            fd.write("          emissiveColor %f %f %f\n" % (self.emissiveColor[0], self.emissiveColor[1], self.emissiveColor[2]))
            fd.write("          specularColor %f %f %f\n" % (self.specularColor[0], self.specularColor[1], self.specularColor[2]))
            fd.write("          ambientIntensity %f\n" % ambInt)
            fd.write("          transparency %f\n" % self.transparency)
            fd.write("          shininess %f\n" % self.shininess)
            fd.write("        }\n      }\n")
            compList.append(matName)
        else:
            fd.write("      appearance Appearance {\n        material USE %s\n      }\n" % matName)


class vrmlTexture(vrmlEntry):
  def __init__(self, parent = None):
    if not isinstance(parent, vrmlAppearance):
      raise Exception()
    vrmlEntry.__init__(self, parent)
    self.texID    = None
    self.fileName = ""
    self.filePath = ""
    self.texType  = None
  def readSpecific(self, fd, string):
    tmp = re.search("url\s+\"([\w\-\.:\/]+)\"", string, re.I | re.S)
    if tmp != None:
      self.fileName = tmp.group(1)
    self.filePath = os.getcwd()
  def __eq__(self, other): #TODO remove in wrlconv
    if not isinstance(other, vrmlTexture):
      return False
    if self.fileName == other.fileName:
      return True
    else:
      return False
  def __ne__(self, other):
    return not self == other


class mesh:
    def __init__(self):
        self.vertexList, self.vertexVBO = None, 0
        self.normalList, self.normalVBO = None, 0
        self.texList, self.texVBO = None, 0
        self.tangentList, self.tangentVBO = None, 0
        self.objects = []
        self.appearance = None
        self.zbuffer = True

    def draw(self):
        glEnableClientState(GL_VERTEX_ARRAY)
        glBindBuffer(GL_ARRAY_BUFFER, self.vertexVBO)
        glVertexPointer(3, GL_FLOAT, 0, None)
        glEnableClientState(GL_NORMAL_ARRAY)
        glBindBuffer(GL_ARRAY_BUFFER, self.normalVBO)
        glNormalPointer(GL_FLOAT, 0, None)
        if self.texList != None:
            glEnableClientState(GL_TEXTURE_COORD_ARRAY)
            glEnableVertexAttribArray(1)
            glBindBuffer(GL_ARRAY_BUFFER, self.texVBO)
            glTexCoordPointer(2, GL_FLOAT, 0, None)
            glBindBuffer(GL_ARRAY_BUFFER, self.tangentVBO)
            glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, None)
        #Disabe writing z-values for transparent objects
        if not self.zbuffer:
            glDepthMask(GL_FALSE)
        for obj in self.objects:
            obj.draw()
        glDisableClientState(GL_TEXTURE_COORD_ARRAY)
        glDisableClientState(GL_NORMAL_ARRAY)
        glDisableClientState(GL_VERTEX_ARRAY)
        glDisableVertexAttribArray(1)
        glDisable(GL_CULL_FACE)
        glDepthMask(GL_TRUE)


class faceset:
    def __init__(self):
        self.mode   = []
        self.index  = []
        self.length = []
        self.solid = False

    def append(self, mode, index, length):
        self.mode.append(mode)
        self.index.append(index)
        self.length.append(length)

    def draw(self):
        if self.solid:
            glEnable(GL_CULL_FACE)
        else:
            glDisable(GL_CULL_FACE)
        for i in range(0, len(self.index)):
            glDrawArrays(self.mode[i], self.index[i], self.length[i])

def loadShader(name):
    fd = open("./shaders/%s.vert" % name, "rb")
    vertShader = fd.read()
    fd.close()
    fd = open("./shaders/%s.frag" % name, "rb")
    fragShader = fd.read()
    fd.close()
    return createShader(vertShader, fragShader)

class render:
    def __init__(self, aScene):
        self.camera = numpy.matrix([[0.], [20.], [20.], [1.]])
        self.pov    = numpy.matrix([[0.], [0.], [0.], [1.]])
        self.lighta = numpy.matrix([[50.], [50.], [50.], [1.]])
        self.lightb = numpy.matrix([[-50.], [-50.], [-50.], [1.]])
        self.axis   = numpy.matrix([[0.], [0.], [1.], [1.]])
        self.updated = True
        self.rotateCamera = False
        self.moveCamera = False
        self.mousePos = [0., 0.]
        self.drawList = None
        self.width = 640
        self.height = 480
        self.cntr = time.time()
        self.fps = 0
        self.data = []
        self.shaders = []
        glutInit(sys.argv)
        glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_ALPHA | GLUT_DEPTH)
        glutInitWindowSize(self.width, self.height)
        glutInitWindowPosition(0, 0)
        glutCreateWindow("VRML viewer")
        glutDisplayFunc(self.drawScene)
        glutIdleFunc(self.drawScene)
        glutReshapeFunc(self.resize)
        glutKeyboardFunc(self.keyHandler)
        glutMotionFunc(self.mouseMove)
        glutMouseFunc(self.mouseButton)
        self.initGraphics()
        self.initScene(aScene)
        glutMainLoop()

    def initGraphics(self):
        glClearColor(0.8, 0.8, 0.8, 0.0)
        glClearDepth(1.0)
        glDepthFunc(GL_LESS)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT1)
        #Setup global lighting
        glLightModelfv(GL_LIGHT_MODEL_AMBIENT, [0.4, 0.4, 0.4, 1.])
        glLightModeli(GL_LIGHT_MODEL_COLOR_CONTROL, GL_SEPARATE_SPECULAR_COLOR)
        #Setup light 0
        glEnable(GL_LIGHT0)
        glLightfv(GL_LIGHT0, GL_POSITION, self.lighta)
        glLightfv(GL_LIGHT0, GL_AMBIENT, [0.0, 0.0, 0.0, 1.])
        glLightfv(GL_LIGHT0, GL_DIFFUSE, [1.0, 1.0, 1.0, 1.])
        glLightfv(GL_LIGHT0, GL_SPECULAR, [0.8, 0.8, 0.8, 1.])
        glLightf(GL_LIGHT0, GL_CONSTANT_ATTENUATION, 1.0)
        glLightf(GL_LIGHT0, GL_LINEAR_ATTENUATION, 0.0)
        glLightf(GL_LIGHT0, GL_QUADRATIC_ATTENUATION, 0.0)
        #Setup light 1
        glEnable(GL_LIGHT1)
        glLightfv(GL_LIGHT1, GL_POSITION, self.lightb)
        glLightfv(GL_LIGHT1, GL_AMBIENT, [0.0, 0.0, 0.0, 1.])
        glLightfv(GL_LIGHT1, GL_DIFFUSE, [1.0, 1.0, 1.0, 1.])
        glLightfv(GL_LIGHT1, GL_SPECULAR, [0.8, 0.8, 0.8, 1.])
        glLightf(GL_LIGHT1, GL_CONSTANT_ATTENUATION, 1.0)
        glLightf(GL_LIGHT1, GL_LINEAR_ATTENUATION, 0.0)
        glLightf(GL_LIGHT1, GL_QUADRATIC_ATTENUATION, 0.0)
        #Blending using shader
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        #glCullFace(GL_BACK)
        self.loadShaders()
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45.0, float(self.width)/float(self.height), 0.1, 1000.0)
        glMatrixMode(GL_MODELVIEW)

    def loadShaders(self):
        oldDir = os.getcwd()
        scriptDir = os.path.dirname(os.path.realpath(__file__))
        if len(scriptDir) > 0:
            os.chdir(scriptDir)
        self.shaders = {}
        self.shaders["colored"] = loadShader("light");
        self.shaders["texture"] = loadShader("light_tex");
        self.shaders["normals"] = loadShader("light_nmap");
        glBindAttribLocation(self.shaders["normals"], 1, "tangentVector")
        glLinkProgram(self.shaders["normals"])
        self.shaders["cubemap"] = loadShader("cubemap");
        os.chdir(oldDir)

    def initScene(self, aScene):
        vrmlShape._vcount = 0
        vrmlShape._pcount = 0
        self.data = aScene.buildMesh()
        print "Total vertex count: %d, polygon count: %d, rendered mesh count: %d" % (vrmlShape._vcount, vrmlShape._pcount, len(self.data))
        for meshEntry in self.data:
            meshEntry.vertexVBO = glGenBuffers(1)
            glBindBuffer(GL_ARRAY_BUFFER, meshEntry.vertexVBO)
            glBufferData(GL_ARRAY_BUFFER, meshEntry.vertexList, GL_STATIC_DRAW)
            #if glIsBuffer(meshEntry.vertexVBO) != GL_FALSE:
              #print "Buffer created"
            meshEntry.normalVBO = glGenBuffers(1)
            glBindBuffer(GL_ARRAY_BUFFER, meshEntry.normalVBO)
            glBufferData(GL_ARRAY_BUFFER, meshEntry.normalList, GL_STATIC_DRAW)
            if meshEntry.texList != None:
                meshEntry.texVBO = glGenBuffers(1)
                glBindBuffer(GL_ARRAY_BUFFER, meshEntry.texVBO)
                glBufferData(GL_ARRAY_BUFFER, meshEntry.texList, GL_STATIC_DRAW)
                meshEntry.tangentVBO = glGenBuffers(1)
                glBindBuffer(GL_ARRAY_BUFFER, meshEntry.tangentVBO)
                glBufferData(GL_ARRAY_BUFFER, meshEntry.tangentList, GL_STATIC_DRAW)
                if meshEntry.appearance != None:
                    for mat in meshEntry.appearance.objects:
                        if isinstance(mat, vrmlTexture) and mat.texID == None:
                            self.loadTexture(mat)

    def setMaterial(self, arg):
        glEnable(GL_COLOR_MATERIAL)
        glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, arg.shininess * 128.)
        glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, arg.specularColor)
        glMaterialfv(GL_FRONT_AND_BACK, GL_EMISSION, arg.emissiveColor)
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT)
        glColor4f(arg.ambientColor[0], arg.ambientColor[1], arg.ambientColor[2], arg.ambientColor[3])
        glColorMaterial(GL_FRONT_AND_BACK, GL_DIFFUSE)
        glColor4f(arg.diffuseColor[0], arg.diffuseColor[1], arg.diffuseColor[2], arg.diffuseColor[3])
        glDisable(GL_COLOR_MATERIAL)

    def loadTexture(self, arg):
        mapPath = re.search("cubemap:(.*?)(\..*)", arg.fileName, re.I)
        if mapPath != None:
            arg.texType = GL_TEXTURE_CUBE_MAP
            arg.texID = glGenTextures(1)
            glBindTexture(GL_TEXTURE_CUBE_MAP, arg.texID)
            glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
            mapTarget = [GL_TEXTURE_CUBE_MAP_POSITIVE_X, GL_TEXTURE_CUBE_MAP_NEGATIVE_X, GL_TEXTURE_CUBE_MAP_NEGATIVE_Y,
                         GL_TEXTURE_CUBE_MAP_POSITIVE_Y, GL_TEXTURE_CUBE_MAP_POSITIVE_Z, GL_TEXTURE_CUBE_MAP_NEGATIVE_Z]
            mapFile   = [arg.filePath + "/" + mapPath.group(1) + "/" + mapPath.group(1) + "_positive_x" + mapPath.group(2),
                         arg.filePath + "/" + mapPath.group(1) + "/" + mapPath.group(1) + "_negative_x" + mapPath.group(2),
                         arg.filePath + "/" + mapPath.group(1) + "/" + mapPath.group(1) + "_positive_y" + mapPath.group(2),
                         arg.filePath + "/" + mapPath.group(1) + "/" + mapPath.group(1) + "_negative_y" + mapPath.group(2),
                         arg.filePath + "/" + mapPath.group(1) + "/" + mapPath.group(1) + "_positive_z" + mapPath.group(2),
                         arg.filePath + "/" + mapPath.group(1) + "/" + mapPath.group(1) + "_negative_z" + mapPath.group(2)]
            print mapFile
            for i in range(0, 6):
                im = Image.open(mapFile[i])
                width, height, image = im.size[0], im.size[1], im.tostring("raw", "RGB", 0, -1)
                #gluBuild2DMipmaps(mapTarget[i], 4, width, height, GL_RGB, GL_UNSIGNED_BYTE, image)
                glTexImage2D(mapTarget[i], 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, image)
        else:
            if not os.path.isfile(arg.filePath + "/" + arg.fileName):
                return
            im = Image.open(arg.filePath + "/" + arg.fileName)
            try:
                #Get image dimensions and data
                width, height, image = im.size[0], im.size[1], im.tostring("raw", "RGBA", 0, -1)
            except SystemError:
                #Has no alpha channel, synthesize one, see the texture module for more realistic handling
                width, height, image = im.size[0], im.size[1], im.tostring("raw", "RGBX", 0, -1)
            arg.texType = GL_TEXTURE_2D
            arg.texID = glGenTextures(1)
            #Make it current
            glBindTexture(GL_TEXTURE_2D, arg.texID)
            glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
            #Copy the texture into the current texture ID
            #glTexImage2D(GL_TEXTURE_2D, 0, 3, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, image)
            #gluBuild2DMipmaps(GL_TEXTURE_2D, GLU_RGBA8, width, height, GL_RGBA, GL_UNSIGNED_BYTE, image)
            gluBuild2DMipmaps(GL_TEXTURE_2D, 4, width, height, GL_RGBA, GL_UNSIGNED_BYTE, image)
        print "Loaded %s, width: %d, height: %d, id: %d" % (arg.name, width, height, arg.texID)

    def setTexture(self, arg, layer = 0):
        texLayer = None
        if layer == 0:
            texLayer = GL_TEXTURE0
        elif layer == 1:
            texLayer = GL_TEXTURE1
        glActiveTexture(texLayer)
        if arg.texType == GL_TEXTURE_2D:
            glEnable(GL_TEXTURE_2D)
            glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            #glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
            glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
            glBindTexture(GL_TEXTURE_2D, arg.texID)
        elif arg.texType == GL_TEXTURE_CUBE_MAP:
            glEnable(GL_TEXTURE_CUBE_MAP)
            glTexParameterf(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            glTexParameterf(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
            #glTexParameterf(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
            glBindTexture(GL_TEXTURE_CUBE_MAP, arg.texID)

    def setAppearance(self, arg):
        texNum = 0
        texType = None
        for mat in arg.objects:
            if isinstance(mat, vrmlMaterial):
                self.setMaterial(mat)
            elif isinstance(mat, vrmlTexture):
                self.setTexture(mat, texNum)
                texType = mat.texType
                texNum += 1
        if texNum == 0:
            glUseProgram(self.shaders["colored"])
        elif texNum == 1:
            if texType == GL_TEXTURE_2D:
                glUseProgram(self.shaders["texture"])
                tex = glGetUniformLocation(self.shaders["texture"], "diffuseTexture")
            else:
                glUseProgram(self.shaders["cubemap"])
                tex = glGetUniformLocation(self.shaders["cubemap"], "diffuseTexture")
            glUniform1i(tex, 0)
        elif texNum >= 2:
            glUseProgram(self.shaders["normals"])
            tex = glGetUniformLocation(self.shaders["normals"], "diffuseTexture")
            glUniform1i(tex, 0)
            tex = glGetUniformLocation(self.shaders["normals"], "normalTexture")
            glUniform1i(tex, 1)

    def drawAxis(self):
        glEnableClientState(GL_VERTEX_ARRAY)
        glEnableClientState(GL_COLOR_ARRAY)
        #glDisable(GL_BLEND)
        #glDisable(GL_COLOR_MATERIAL)
        glEnable(GL_COLOR_MATERIAL)
        glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, 0.)
        glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, [0., 0., 0., 1.])
        glMaterialfv(GL_FRONT_AND_BACK, GL_EMISSION, [0., 0., 0., 1.])
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
        glColor4f(1., 1., 1., 1.)
        varray = numpy.array([0.0, 0.0, 0.0,
                              4.0, 0.0, 0.0,
                              0.0, 0.0, 0.0,
                              0.0, 4.0, 0.0,
                              0.0, 0.0, 0.0,
                              0.0, 0.0, 2.0], dtype=numpy.float32)
        carray = numpy.array([1.0, 0.0, 0.0, 1.0,
                              1.0, 0.0, 0.0, 1.0,
                              0.0, 1.0, 0.0, 1.0,
                              0.0, 1.0, 0.0, 1.0,
                              0.0, 0.0, 1.0, 1.0,
                              0.0, 0.0, 1.0, 1.0], dtype=numpy.float32)
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glColorPointer(4, GL_FLOAT, 0, carray)
        glVertexPointer(3, GL_FLOAT, 0, varray)
        #glEnable(GL_COLOR_MATERIAL)
        glDrawArrays(GL_LINES, 0, len(varray) / 3)
        glDisable(GL_COLOR_MATERIAL)
        #glEnable(GL_BLEND)
        glDisableClientState(GL_COLOR_ARRAY)
        glDisableClientState(GL_VERTEX_ARRAY)

    def drawScene(self):
        self.updated = True
        if self.updated:
            self.updated = False
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            glLoadIdentity()
            gluLookAt(float(self.camera[0]), float(self.camera[1]), float(self.camera[2]), 
                      float(self.pov[0]), float(self.pov[1]), float(self.pov[2]), 
                      float(self.axis[0]), float(self.axis[1]), float(self.axis[2]))
            glLightfv(GL_LIGHT0, GL_POSITION, self.lighta)
            glLightfv(GL_LIGHT1, GL_POSITION, self.lightb)
            glUseProgram(0)
            self.drawAxis()
            for current in self.data:
                self.setAppearance(current.appearance)
                current.draw()
                glDisable(GL_TEXTURE_2D)
            glutSwapBuffers()
            self.fps += 1
            if time.time() - self.cntr >= 1.:
                glutSetWindowTitle("VRML viewer: %d vertices, %d polygons, %d FPS" % (vrmlShape._vcount, vrmlShape._pcount, self.fps / (time.time() - self.cntr)))
                self.cntr = time.time()
                self.fps = 0
        else:
            time.sleep(.001)

    def resize(self, width, height):
        if height == 0:
            height = 1
        self.width = width
        self.height = height
        glViewport(0, 0, self.width, self.height)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45.0, float(self.width)/float(self.height), 0.1, 1000.0)
        glMatrixMode(GL_MODELVIEW)
        self.updated = True

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
            scaleMatrix = numpy.matrix([[zm, 0., 0., 0.],
                                        [0., zm, 0., 0.],
                                        [0., 0., zm, 0.],
                                        [0., 0., 0., 1.]])
            self.camera -= self.pov
            self.camera = scaleMatrix * self.camera
            self.camera += self.pov
        elif bNumber == 4 and bAction == GLUT_DOWN:
            zm = 1.1
            scaleMatrix = numpy.matrix([[zm, 0., 0., 0.],
                                        [0., zm, 0., 0.],
                                        [0., 0., zm, 0.],
                                        [0., 0., 0., 1.]])
            self.camera -= self.pov
            self.camera = scaleMatrix * self.camera
            self.camera += self.pov
        self.updated = True
    def mouseMove(self, xPos, yPos):
        if self.rotateCamera:
            self.camera -= self.pov
            zrot = (self.mousePos[0] - xPos) / 100.
            nrot = (yPos - self.mousePos[1]) / 100.
            if zrot != 0.:
                rotMatrixA = numpy.matrix([[math.cos(zrot), -math.sin(zrot), 0., 0.],
                                           [math.sin(zrot),  math.cos(zrot), 0., 0.],
                                           [            0.,              0., 1., 0.],
                                           [            0.,              0., 0., 1.]])
                self.camera = rotMatrixA * self.camera
            if nrot != 0.:
                normal = normalize(getNormal(self.camera, self.axis))
                angle = getAngle(self.camera, self.axis)
                if (nrot > 0 and nrot > angle) or (nrot < 0 and -nrot > math.pi - angle):
                    self.axis = -self.axis
                rotMatrixB = fillRotateMatrix(normal, nrot)
                self.camera = rotMatrixB * self.camera
            self.camera += self.pov
            self.mousePos = [xPos, yPos]
        elif self.moveCamera:
            tlVector = numpy.matrix([[(xPos - self.mousePos[0]) / 50.], [(self.mousePos[1] - yPos) / 50.], [0.], [0.]])
            self.camera -= self.pov
            normal = normalize(getNormal([0., 0., 1.], self.camera))
            angle = getAngle(self.camera, [0., 0., 1.])
            ah = getAngle(normal, [1., 0., 0.])
            rotZ = numpy.matrix([[math.cos(ah), -math.sin(ah), 0., 0.],
                                 [math.sin(ah),  math.cos(ah), 0., 0.],
                                 [          0.,            0., 1., 0.],
                                 [          0.,            0., 0., 1.]])
            self.camera += self.pov
            rotCNormal = fillRotateMatrix(normal, angle)
            tlVector = rotZ * tlVector
            tlVector = rotCNormal * tlVector
            self.camera = self.camera - tlVector
            self.pov = self.pov - tlVector
            self.mousePos = [xPos, yPos]
        self.updated = True
    def keyHandler(self, key, xPos, yPos):
        if key in ("\x1b", "q", "Q"):
            exit()
        if key in ("r", "R"):
            self.camera = numpy.matrix([[0.], [20.], [0.], [1.]])
            self.pov    = numpy.matrix([[0.], [0.], [0.], [1.]])
        #if key in ("w", "W"): #FIXME rewrite
            #vect = normalize(self.pov - self.camera)
            #self.pov += vect
            #self.camera += vect
        #if key in ("s", "S"):
            #vect = normalize(self.pov - self.camera)
            #vect[3] = 0.
            #self.pov -= vect
            #self.camera -= vect
        #if key in ("a", "A"):
            #normal = normalize(getNormal([0., 0., 1.], self.camera - self.pov))
            #normal = numpy.matrix([[float(normal[0])], [float(normal[1])], [float(normal[2])], [0.]])
            #self.pov -= normal
            #self.camera -= normal
        #if key in ("d", "D"):
            #normal = normalize(getNormal([0., 0., 1.], self.camera - self.pov))
            #normal = numpy.matrix([[float(normal[0])], [float(normal[1])], [float(normal[2])], [0.]])
            #self.pov += normal
            #self.camera += normal
        self.updated = True

parser = argparse.ArgumentParser()
parser.add_argument("-v", dest="view", help="render model", default=False, action="store_true")
parser.add_argument("-w", dest="rebuild", help="rebuild model", default=False, action="store_true")
parser.add_argument("-t", dest="translate", help="move mesh to new coordinates [x,y,z], default value \"0.,0.,0.\"", default='0.,0.,0.')
parser.add_argument("-r", dest="rotate", help="rotate mesh around vector [x,y,z] by angle in degrees, default value \"0.,0.,1.,0.\"", default='0.,0.,1.,0.')
parser.add_argument("-s", dest="scale", help="scale shapes by [x,y,z], default value \"1.,1.,1.\"", default='1.,1.,1.')
parser.add_argument("-f", dest="pattern", help="regular expression, filter objects by name", default="")
parser.add_argument("-d", dest="debug", help="show debug information", default=False, action="store_true")
parser.add_argument(dest="files", nargs="*")
options = parser.parse_args()

debug = options.debug

options.translate = options.translate.split(",")
options.rotate = options.rotate.split(",")
options.scale = options.scale.split(",")
gTranslate = [0., 0., 0.]
gRotate = [0., 0., 1., 0.]
gScale = [1., 1., 1.]
for i in range(0, 3):
    try:
        gTranslate[i] = float(options.translate[i])
    except ValueError:
        print "Wrong translate parameter: %s" % options.translate[i]
        exit()
    try:
        gScale[i] = float(options.scale[i])
    except ValueError:
        print "Wrong scale parameter: %s" % options.scale[i]
        exit()
for i in range(0, 4):
    try:
        gRotate[i] = float(options.rotate[i])
    except ValueError:
        print "Wrong rotate parameter: %s" % options.rotate[i]
        exit()

gRotate[3] *= (math.pi / 180)
sc = vrmlScene()
sc.setTransform(gTranslate, gRotate, gScale)

x3dSupport = True
x3dScript = "X3dToVrml97.xslt"
try:
    nf = open(os.devnull, "w")
    subprocess.call(["xmlto", "--version"], stdout = nf)
    nf.close()
    scriptDir = os.path.dirname(os.path.realpath(__file__))
    if len(scriptDir) > 0:
        x3dScript = scriptDir + "/" + x3dScript
except:
    x3dSupport = False

for fileName in options.files:
    if os.path.isfile(fileName):
        if os.path.splitext(fileName)[1] == ".x3d":
            if not x3dSupport:
                print "Install package \"xmlto\" to handle X3D files"
                exit()
            procFile = os.path.splitext(fileName)[0] + ".proc"
            modelTime, procTime = 0.0, 0.0
            procExist = os.path.isfile(procFile)
            if procExist:
                modelTime = os.path.getctime(fileName)
                procTime = os.path.getctime(procFile)
                pDebug("X3D model time: %s, VRML model time: %s" % (time.ctime(modelTime), time.ctime(procTime)))
            if not procExist or modelTime >= procTime:
                nf = open(os.devnull, "w")
                subprocess.call(["xmlto", "-x", x3dScript, "html", "--skip-validation", fileName], stderr = nf)
                nf.close()
            if os.path.isfile(procFile):
                sc.loadFile(procFile, options.pattern)
        else:
            sc.loadFile(fileName, options.pattern)
        if options.rebuild:
            rebuilt = os.path.splitext(fileName)[0] + ".re.wrl"
            sc.saveFile(rebuilt)
            pDebug("Input file %s saved to %s" % (fileName, rebuilt))

if options.view:
    if not opengl:
        print "Please install PyOpenGL package"
    else:
        rend = render(sc)

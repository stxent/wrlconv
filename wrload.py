#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: xen (alexdmitv@gmail.com)
# License: Public domain code
import re
import math
import numpy
import sys
import time
import optparse
import copy
import Image
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *

def getAngle(v1, v2):
  mag1 = math.sqrt(v1[0]*v1[0] + v1[1]*v1[1] + v1[2]*v1[2])
  mag2 = math.sqrt(v2[0]*v2[0] + v2[1]*v2[1] + v2[2]*v2[2])
  res = (v1[0]*v2[0] + v1[1]*v2[1] + v1[2]*v2[2]) / (mag1 * mag2)
  ac = math.acos(res)
  if v2[0]*v1[1] - v2[1]*v1[0] < 0:
    ac *= -1
  return ac

def getNormal(v1, v2):
  return numpy.matrix([[float(v1[1] * v2[2] - v1[2] * v2[1])],
                       [float(v1[2] * v2[0] - v1[0] * v2[2])],
                       [float(v1[0] * v2[1] - v1[1] * v2[0])]])

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
      if data[i] == '{' or data[i] == '[':
        balance += 1
      if data[i] == '}' or data[i] == ']':
        balance -= 1
      if balance == 0:
        return len(data) - i - 2
  return 0

def calcBalance(string, delta = None, openset = ('[', '{'), closeset = (']', '}')):
  balance, offset, update = 0, 0, False
  for i in range(0, len(string)):
    if string[i] in openset:
      balance += 1
      #update = True
      update = False
      #if delta == None:
        #offset = len(string) - i - 1
    if string[i] in closeset:
      balance -= 1
      update = True
      #if delta == None:
        #offset = len(string) - i - 1
    #if delta != None and offset == 0 and balance >= delta:
    if update == True and delta != None and balance >= delta:
      offset = len(string) - i - 1
      #print "Offset: %d Index: %d" % (offset, i)
      update = False
  return (balance, offset)

def fillString(length):
  res = ""
  while length > 0:
    res += " "
    length -= 1
  return res

class vrmlEntry:
  def __init__(self, parent = None):
    self.parent = parent
    self.name = ""
    self.objects = []
    if self.parent != None:
      self._level = self.parent._level + 2
    else:
      self._level = 0
  def read(self, fd):
    defPattern = re.compile("([\w]*?)\s*([\w\-]*?)\s*(\w+)\s*{", re.I | re.S)
    delta, offset, balance = 0, 0, 0
    #Highest level
    while 1:
      data = fd.readline()
      if len(data) == 0:
        break
      #print "%s - %d: '%s'" % (self.__class__.__name__, balance, data.replace("\n", "").replace("\r", ""))
      regexp = defPattern.search(data)
      if regexp != None:
        (delta, offset) = calcBalance(data[:regexp.start()], -1, ('{'), ('}'))
        balance += delta
        initialPos = fd.tell()
        self.readSpecific(fd, data[:regexp.start()]) #FIXME added
        if initialPos != fd.tell():
          print "%sRead error" % fillString(self._level)
          break
        if balance < 0:
          print "%sWrong balance: %d" % (fillString(self._level), balance)
          fd.seek(-(len(data) - regexp.start() + offset), os.SEEK_CUR)
          break
        fd.seek(-(len(data) - regexp.end()), os.SEEK_CUR)
        entry = None
        #print "%sBalstring: '%s'" % (fillString(self._level), data[:regexp.start()])
        print "%sEntry: '%s' '%s' '%s' Balance: %d" % (fillString(self._level), regexp.group(1), regexp.group(2), regexp.group(3), balance)
        try:
          if regexp.group(3) == "Transform" or regexp.group(3) == "Group":
            entry = vrmlTransform(self)
          #elif regexp.group(3) == "Group":
            #self.read(fd)
          elif regexp.group(3) == "Appearance":
            entry = vrmlAppearance(self)
          elif regexp.group(3) == "Shape":
            entry = vrmlShape(self)
          elif regexp.group(3) == "Material":
            entry = vrmlMaterial(self)
          elif regexp.group(3) == "ImageTexture":
            entry = vrmlTexture(self)
          elif regexp.group(3) == "IndexedFaceSet":
            entry = vrmlGeometry(self)
          elif regexp.group(3) == "Coordinate":
            entry = vrmlCoordinates(self, 'model')
          elif regexp.group(3) == "TextureCoordinate":
            entry = vrmlCoordinates(self, 'texture')
          elif regexp.group(3) == "Inline":
            entry = vrmlInline(self)
          else:
            offset = skipChunk(fd)
            fd.seek(-offset, os.SEEK_CUR)
        except:
          print "%sChunk sequence error" % fillString(self._level)
          pass
        if entry != None:
          if regexp.group(1) == "DEF" and len(regexp.group(2)) > 0:
            entry.name = regexp.group(2)
          entry.read(fd)
          self.objects.append(entry)
          ptr = self
          while isinstance(ptr, vrmlScene) == False:
            ptr = ptr.parent
          ptr.entries.append(entry)
        #balance -= delta
      else:
        (delta, offset) = calcBalance(data, -(balance + 1), ('{'), ('}'))
        #print "B: %d B+: %d Off: %d Left: '%s' from '%s'" % (balance, -(balance + 1), offset, data[:len(data) - offset].replace("\n", "").replace("\r", "").replace("\t", " "), data.replace("\n", "").replace("\r", "").replace("\t", " "))
        balance += delta
        #print "SEARCH: '%s'" % data[:len(data) - offset].replace("\n", "").replace("\r", "")
        initialPos = fd.tell()
        self.readSpecific(fd, data)
        using = re.search("USE\s+([\w\-]+)", data, re.I | re.S)
        if using != None and using.start() < len(data) - offset:
          print "%sUsing entry %s" % (fillString(self._level), using.group(1))
          ptr = self
          while not isinstance(ptr, vrmlScene) and not isinstance(ptr, vrmlInline):
            ptr = ptr.parent
          for obj in ptr.entries:
            if obj.name == using.group(1):
              print "%sFound entry %s" % (fillString(self._level), using.group(1))
              self.objects.append(obj)
        if balance < 0:
          print "%sBalance error: %d" % (fillString(self._level), balance)
          if initialPos == fd.tell():
            fd.seek(-offset, os.SEEK_CUR) #FIXME HIGH PRIORITY
          break
  def readSpecific(self, fd, string):
    #print "Class: %s" % self.__class__.__name__
    #print "Reading spec: %s" % string.replace("\n", "").replace("\t", "")
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
    #self.transform = translation + rotation * scale
    self.transform = translation + rotation * scale
  def loadFile(self, fileName):
    wrlFile = open(fileName, "r")
    oldDir = os.getcwd()
    if len(os.path.dirname(fileName)) > 0:
      os.chdir(os.path.dirname(fileName))
    self.read(wrlFile)
    os.chdir(oldDir)
    wrlFile.close()
  def saveFile(self, fileName):
    wrlFile = open(fileName, "w")
    compList = []
    wrlFile.write("#VRML V2.0 utf8\n#Exported from Blender by wrlconv.py\n")
    for entry in self.objects:
      if isinstance(entry, vrmlShape) or isinstance(entry, vrmlTransform):
        entry.write(wrlFile, compList, self.transform)
    wrlFile.close()

class vrmlInline(vrmlEntry): #FIXME ALL
  def __init__(self, parent):
    vrmlEntry.__init__(self, parent)
    self.entries = []
  def readSpecific(self, fd, string):
    urlSearch = re.search("url\s+\"([\w\-\._\/]+)\"", string, re.S)
    if urlSearch != None:
      oldDir = os.getcwd()
      if os.path.isfile(urlSearch.group(1)):
        #print "%sLoading file: %s" % (fillString(self._level), urlSearch.group(1))
        #self.name = urlSearch.group(1)
        wrlFile = open(urlSearch.group(1), "r")
        if len(os.path.dirname(urlSearch.group(1))) > 0:
          os.chdir(os.path.dirname(urlSearch.group(1)))
        self.read(wrlFile)
        wrlFile.close()
      else:
        print "%sFile not found: %s" % (fillString(self._level), urlSearch.group(1))
      os.chdir(oldDir)
  def mesh(self, transform, _offset = 0):
    print "%sDraw inline: %s" % (fillString(_offset), self.name)
    res = []
    for obj in self.objects:
      #print "%sSubobject: %s (%s)" % (fillString(_offset), obj.name, obj.__class__.__name__)
      if isinstance(obj, vrmlShape) or isinstance(obj, vrmlTransform) or isinstance(obj, vrmlInline):
        res.extend(obj.mesh(transform, _offset + 2))
    return res
  def write(self, fd, compList, transform):
    for obj in self.objects:
      if isinstance(obj, vrmlShape) or isinstance(obj, vrmlTransform) or isinstance(obj, vrmlInline):
        obj.write(fd, compList, transform)

class vrmlTransform(vrmlEntry):
  def __init__(self, parent):
    vrmlEntry.__init__(self, parent)
    self.transform = numpy.matrix([[1., 0., 0., 0.],
                                   [0., 1., 0., 0.],
                                   [0., 0., 1., 0.],
                                   [0., 0., 0., 1.]])
  def readSpecific(self, fd, string):
    #print "read trans: %s" % string
    tmp = re.search("translation\s+([+e\d\.\-]+)\s+([+e\d\.\-]+)\s+([+e\d\.\-]+)", string, re.I | re.S)
    if tmp != None:
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
    if tmp != None:
      tform = fillRotateMatrix([float(tmp.group(1)), float(tmp.group(2)), float(tmp.group(3))], float(tmp.group(4)))
      self.transform = self.transform * tform
    tmp = re.search("scale\s+([+e\d\.\-]+)\s+([+e\d\.\-]+)\s+([+e\d\.\-]+)", string, re.I | re.S)
    if tmp != None:
      tform = numpy.matrix([[float(tmp.group(1)), 0., 0., 0.],
                            [0., float(tmp.group(2)), 0., 0.],
                            [0., 0., float(tmp.group(3)), 0.],
                            [0., 0., 0., 1.]])
      self.transform = self.transform * tform
  def mesh(self, transform, _offset = 0):
    print "%sDrawing transform: %s" % (fillString(_offset), self.name)
    res = []
    tform = transform * self.transform
    for obj in self.objects:
      #print "%sSubobject: %s (%s)" % (fillString(_offset), obj.name, obj.__class__.__name__)
      if isinstance(obj, vrmlShape) or isinstance(obj, vrmlTransform) or isinstance(obj, vrmlInline):
        res.extend(obj.mesh(tform, _offset + 2))
    return res
  def write(self, fd, compList, transform):
    tform = transform * self.transform
    for obj in self.objects:
      if isinstance(obj, vrmlShape) or isinstance(obj, vrmlTransform) or isinstance(obj, vrmlInline):
        obj.write(fd, compList, tform)

class vrmlShape(vrmlEntry):
  _vcount = 0
  _pcount = 0
  def __init__(self, parent):
    vrmlEntry.__init__(self, parent)
  def mesh(self, transform, _offset = 0):
    print "%sDraw shape %s" % (fillString(_offset), self.name)
    newMesh = mesh()
    for obj in self.objects:
      if isinstance(obj, vrmlAppearance):
        for app in obj.objects:
          if isinstance(app, vrmlMaterial) or isinstance(app, vrmlTexture):
            newMesh.materials.append(app)
      elif isinstance(obj, vrmlGeometry):
        _tsa = time.time()
        print "%sDraw geometry %s" % (fillString(_offset + 2), obj.name)
        newMesh.solid = obj.solid
        if obj.polygonsUV != None and len(obj.polygons) == len(obj.polygonsUV):
          genTex = True
        else:
          genTex = False
        tmpVertices = []
        tmpVerticesUV = []
        for coords in obj.objects:
          if isinstance(coords, vrmlCoordinates) and coords.cType == vrmlCoordinates.TYPE['model']:
            for vert in coords.vertices:
              tmp = numpy.matrix([[vert[0]], [vert[1]], [vert[2]], [1.]])
              tmp = transform * tmp
              tmpVertices.append(numpy.array([float(tmp[0]), float(tmp[1]), float(tmp[2])]))
          elif isinstance(coords, vrmlCoordinates) and coords.cType == vrmlCoordinates.TYPE['texture']:
            for vert in coords.vertices:
              tmpVerticesUV.append(vert)
        if obj.triCount > 0:
          fsTri = faceset(GL_TRIANGLES)
          fsTri.append(0, obj.triCount)
          newMesh.objects.append(fsTri)
        if obj.quadCount > 0:
          fsQuad = faceset(GL_QUADS)
          fsQuad.append(0, obj.quadCount)
          newMesh.objects.append(fsQuad)
        if obj.polyCount > 0:
          fsPoly = faceset(GL_POLYGON)
        else:
          fsPoly = None
        length = obj.triCount + obj.quadCount + obj.polyCount
        #if obj.smooth == False:
          ##Normal facesets
          #fsNorm = faceset(GL_LINES)
          #fsNorm.append(length, length * 2)
          #newMesh.objects.append(fsNorm)
          #length += length * 2
        #else:
          ##Normal facesets
          #fsNorm = faceset(GL_LINES)
          #fsNorm.append(length, len(tmpVertices) * 2)
          #newMesh.objects.append(fsNorm)
          #length += len(tmpVertices) * 2
        newMesh.vertexList = numpy.zeros(length * 3, dtype = numpy.float32)
        newMesh.normalList = numpy.zeros(length * 3, dtype = numpy.float32)
        if genTex == True:
          newMesh.texList = numpy.zeros(length * 2, dtype = numpy.float32)
        tPos = 0
        qPos = obj.triCount
        pPos = obj.triCount + obj.quadCount
        ##Normal start index
        #nPos = obj.triCount + obj.quadCount + obj.polyCount
        if obj.smooth == False:
          for poly in range(0, len(obj.polygons)):
            normal = getNormal(tmpVertices[obj.polygons[poly][1]] - tmpVertices[obj.polygons[poly][0]], 
                               tmpVertices[obj.polygons[poly][0]] - tmpVertices[obj.polygons[poly][2]])
            det = numpy.linalg.norm(normal)
            if det != 0:
              normal /= -det
            pos = 0
            if len(obj.polygons[poly]) == 3:
              pos = tPos
              tPos += 3
            elif len(obj.polygons[poly]) == 4:
              pos = qPos
              qPos += 4
            else:
              fsPoly.append(pPos, len(obj.polygons[poly]))
              pos = pPos
              pPos += len(obj.polygons[poly])
            for ind in range(0, len(obj.polygons[poly])):
              newMesh.vertexList[3 * pos]     = tmpVertices[obj.polygons[poly][ind]][0]
              newMesh.normalList[3 * pos]     = normal[0]
              newMesh.vertexList[3 * pos + 1] = tmpVertices[obj.polygons[poly][ind]][1]
              newMesh.normalList[3 * pos + 1] = normal[1]
              newMesh.vertexList[3 * pos + 2] = tmpVertices[obj.polygons[poly][ind]][2]
              newMesh.normalList[3 * pos + 2] = normal[2]
              ##Draw normal
              #normVert = numpy.array([tmpVertices[obj.polygons[poly][ind]][0] + normal[0] / 4, 
                                      #tmpVertices[obj.polygons[poly][ind]][1] + normal[1] / 4, 
                                      #tmpVertices[obj.polygons[poly][ind]][2] + normal[2] / 4])
              #newMesh.vertexList[3 * nPos]     = tmpVertices[obj.polygons[poly][ind]][0]
              #newMesh.vertexList[3 * nPos + 1] = tmpVertices[obj.polygons[poly][ind]][1]
              #newMesh.vertexList[3 * nPos + 2] = tmpVertices[obj.polygons[poly][ind]][2]
              #newMesh.vertexList[3 * nPos + 3] = normVert[0]
              #newMesh.vertexList[3 * nPos + 4] = normVert[1]
              #newMesh.vertexList[3 * nPos + 5] = normVert[2]
              #nPos += 2
              if genTex == True:
                newMesh.texList[2 * pos]     = tmpVerticesUV[obj.polygonsUV[poly][ind]][0]
                newMesh.texList[2 * pos + 1] = tmpVerticesUV[obj.polygonsUV[poly][ind]][1]
              pos += 1
        else:
          tmpNormals = []
          for i in range(0, len(tmpVertices)):
            tmpNormals.append(numpy.array([0., 0., 0.,]))
          for poly in obj.polygons:
            normal = getNormal(tmpVertices[poly[1]] - tmpVertices[poly[0]], tmpVertices[poly[0]] - tmpVertices[poly[2]])
            det = numpy.linalg.norm(normal)
            if det != 0:
              normal /= -det
            for ind in poly:
              tmpNormals[ind] += numpy.array([float(normal[0]), float(normal[1]), float(normal[2])])
          for i in range(0, len(tmpNormals)):
            det = numpy.linalg.norm(tmpNormals[i])
            if det != 0:
              tmpNormals[i] /= det
            #normVert = tmpVertices[i] + tmpNormals[i] / 4
            ##Draw normal
            #newMesh.vertexList[3 * nPos]     = tmpVertices[i][0]
            #newMesh.vertexList[3 * nPos + 1] = tmpVertices[i][1]
            #newMesh.vertexList[3 * nPos + 2] = tmpVertices[i][2]
            #newMesh.vertexList[3 * nPos + 3] = normVert[0]
            #newMesh.vertexList[3 * nPos + 4] = normVert[1]
            #newMesh.vertexList[3 * nPos + 5] = normVert[2]
            #nPos += 2
          for poly in range(0, len(obj.polygons)):
            pos = 0
            if len(obj.polygons[poly]) == 3:
              pos = tPos
              tPos += 3
            elif len(obj.polygons[poly]) == 4:
              pos = qPos
              qPos += 4
            else:
              fsPoly.append(pPos, len(obj.polygons[poly]))
              pos = pPos
              pPos += len(obj.polygons[poly])
            for ind in range(0, len(obj.polygons[poly])):
              newMesh.vertexList[3 * pos]     = tmpVertices[obj.polygons[poly][ind]][0]
              newMesh.normalList[3 * pos]     = tmpNormals[obj.polygons[poly][ind]][0]
              newMesh.vertexList[3 * pos + 1] = tmpVertices[obj.polygons[poly][ind]][1]
              newMesh.normalList[3 * pos + 1] = tmpNormals[obj.polygons[poly][ind]][1]
              newMesh.vertexList[3 * pos + 2] = tmpVertices[obj.polygons[poly][ind]][2]
              newMesh.normalList[3 * pos + 2] = tmpNormals[obj.polygons[poly][ind]][2]
              if genTex == True:
                newMesh.texList[2 * pos]     = tmpVerticesUV[obj.polygonsUV[poly][ind]][0]
                newMesh.texList[2 * pos + 1] = tmpVerticesUV[obj.polygonsUV[poly][ind]][1]
              pos += 1
        _tsb = time.time()
        if fsPoly:
          newMesh.objects.append(fsPoly)
        vrmlShape._vcount += len(newMesh.vertexList) / 3
        vrmlShape._pcount += len(obj.polygons)
        print "%sCreated in: %f, vertices: %d, polygons: %d" % (fillString(_offset + 2), _tsb - _tsa, len(newMesh.vertexList) / 3, len(obj.polygons))
    return [newMesh]
  def write(self, fd, compList, transform):
    print "Write object %s" % self.name
    if self.parent and self.parent.name != "":
      fd.write("DEF %s Transform {\n  children [\n" % self.parent.name)
    else:
      fd.write("Transform {\n  children [\n")
    fd.write("    Shape {\n")
    for obj in self.objects:
      if isinstance(obj, vrmlAppearance):
        for mat in obj.objects:
          if isinstance(mat, vrmlMaterial):
            mat.write(fd, compList)
      elif isinstance(obj, vrmlGeometry):
        fd.write("      geometry IndexedFaceSet {\n        coord Coordinate { point [\n")
        for coords in obj.objects:
          if isinstance(coords, vrmlCoordinates) and coords.cType == vrmlCoordinates.TYPE['model']:
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
    self.polyCount  = 0
    self.polygonsUV = None
  def readSpecific(self, fd, string):
    #print "%sTry geo read: %s" % (fillString(self._level), string.replace("\n", "").replace("\t", ""))
    #deltaMain = 0
    initialPos = fd.tell()
    paramSearch = re.search("solid\s+(TRUE|FALSE)", string, re.S)
    if paramSearch != None:
      #print "SOLIDITY: %s" % paramSearch.group(1)
      if paramSearch.group(1) == "TRUE":
        self.solid = True
      #else:
        #self.solid = False
    coordSearch = re.search("coordIndex\s*\[", string, re.S)
    texSearch = re.search("texCoordIndex\s*\[", string, re.S)
    if coordSearch != None or texSearch != None:
      #if coordSearch != None:
        #fd.seek(-(len(string) - coordSearch.end()), os.SEEK_CUR) #FIXME
      #elif texSearch != None:
        #fd.seek(-(len(string) - texSearch.end()), os.SEEK_CUR) #FIXME
      print "%sStart polygon read" % fillString(self._level)
      polyPattern = re.compile("([ ,\t\d]+)-1", re.I | re.S)
      indPattern = re.compile("[ ,\t]*(\d+)[ ,\t]*", re.I | re.S)
      tmpPolygons = []
      delta, offset, balance = 0, 0, 0
      data = string
      if coordSearch != None:
        pPos = coordSearch.end()
      elif texSearch != None:
        pPos = texSearch.end()
      #pPos = indexSearch.end()
      while 1:
        #data = fd.readline()
        #print data
        #if len(data) == 0:
          #break
        #pPos = 0
        while 1:
          regexp = polyPattern.search(data, pPos)
          if regexp != None:
            (delta, offset) = calcBalance(data[pPos:regexp.start()], -1, (), (']'))
            balance += delta
            offset = len(data) - regexp.start() + offset
            if balance != 0:
              print "%sWrong balance: %d, offset: %d" % (fillString(self._level), balance, offset)
              break
            polyData = []
            indPos = 0
            while 1:
              ind = indPattern.search(regexp.group(1), indPos)
              if ind == None:
                break
              polyData.append(int(ind.group(1)))
              indPos = ind.end()
            if coordSearch != None:
              if len(polyData) == 3:
                self.triCount += 3
              elif len(polyData) == 4:
                self.quadCount += 4
              else:
                self.polyCount += len(polyData)
            tmpPolygons.append(polyData)
            pPos = regexp.end()
          else:
            (delta, offset) = calcBalance(data, None, (), (']'))
            balance += delta
            offset = len(data)
            break
        if balance != 0:
          if initialPos != fd.tell():
            fd.seek(-offset, os.SEEK_CUR)
          print "%sBalance error: %d, offset: %d" % (fillString(self._level), balance, offset)
          break
        data = fd.readline()
        #deltaMain = 1
        if len(data) == 0:
          break
        pPos = 0
      if coordSearch != None:
        self.polygons = tmpPolygons
        print "%sRead poly done, %d tri, %d quad, %d poly, %d vertices" % (fillString(self._level), self.triCount / 3, self.quadCount / 4, 
                                                                           len(self.polygons) - self.triCount / 3 - self.quadCount / 4,
                                                                           self.polyCount + self.triCount + self.quadCount)
      elif texSearch != None:
        self.polygonsUV = tmpPolygons
        print "%sRead UV poly done, %d poly, %d vertices" % (fillString(self._level), len(self.polygonsUV), 0)

class vrmlCoordinates(vrmlEntry):
  TYPE = {'model' : 0, 'texture' : 1}
  def __init__(self, parent, cType):
    if not isinstance(parent, vrmlGeometry):
      raise Exception()
    vrmlEntry.__init__(self, parent)
    self.cType = vrmlCoordinates.TYPE[cType]
    self.vertices = None
  def readSpecific(self, fd, string):
    #deltaMain = 0
    initialPos = fd.tell()
    #print "%sTry coord read: %s, type: %d" % (fillString(self._level), string.replace("\n", "").replace("\t", ""), self.cType)
    indexSearch = re.search("point\s*\[", string, re.S)
    if indexSearch != None:
      #fd.seek(-(len(string) - indexSearch.end()), os.SEEK_CUR) #FIXME
      print "%sStart vertex read, type: %s" % (fillString(self._level), vrmlCoordinates.TYPE.keys()[self.cType])
      if self.cType == vrmlCoordinates.TYPE['model']:
        vertexPattern = re.compile("([+e\d\-\.]+)[ ,\t]+([+e\d\-\.]+)[ ,\t]+([+e\d\-\.]+)", re.I | re.S)
      elif self.cType == vrmlCoordinates.TYPE['texture']:
        vertexPattern = re.compile("([+e\d\-\.]+)[ ,\t]+([+e\d\-\.]+)", re.I | re.S)
      self.vertices = []
      delta, offset, balance = 0, 0, 0
      data = string
      vPos = indexSearch.end()
      while 1:
        #print "Balance: %d, str: '%s'" % (balance, data.replace("\n", ""))
        while 1:
          regexp = vertexPattern.search(data, vPos)
          if regexp != None:
            #print "SAGA: '%s'" % data[vPos:regexp.start()].replace("\n", "").replace("\t", "")
            #(delta, offset) = calcBalance(data[vPos:regexp.start()], -1, (), (']', '}'))
            (delta, offset) = calcBalance(data[vPos:regexp.start()], -1, (), ('}'))
            balance += delta
            offset = len(data) - regexp.start() + offset
            #if deltaMain:
            if initialPos != fd.tell():
              offset += 1
            #print "LEFT: %s" % data[len(data) - offset:]
            #print "%s %s %s" % (regexp.group(1), regexp.group(2), regexp.group(3))
            if balance != 0:
              #offset += 1
              print "%sWrong balance: %d, offset: %d" % (fillString(self._level), balance, offset)
              break
            if self.cType == vrmlCoordinates.TYPE['model']:
              self.vertices.append(numpy.array([float(regexp.group(1)), float(regexp.group(2)), float(regexp.group(3))]))
            elif self.cType == vrmlCoordinates.TYPE['texture']:
              self.vertices.append(numpy.array([float(regexp.group(1)), float(regexp.group(2))]))
            vPos = regexp.end()
          else:
            #(delta, offset) = calcBalance(data[vPos:], -1, (), (']', '}'))
            (delta, offset) = calcBalance(data[vPos:], -1, (), ('}'))
            balance += delta
            #if deltaMain:
            if initialPos != fd.tell():
              offset += 1
            #offset += 1
            break
        if balance != 0:
          if initialPos != fd.tell():
            fd.seek(-offset, os.SEEK_CUR)
          #print "STR: str: '%s'" % (data[len(data) - offset:].replace("\n", ""))
          print "%sBalance error: %d, offset: %d" % (fillString(self._level), balance, offset)
          break
        data = fd.readline()
        #deltaMain = 1
        if len(data) == 0:
          break
        vPos = 0
      print "%sEnd vertex read, count: %d" % (fillString(self._level), len(self.vertices))

class vrmlAppearance(vrmlEntry):
  def __init__(self, parent = None):
    vrmlEntry.__init__(self, parent)

class vrmlMaterial(vrmlEntry):
  def __init__(self, parent = None):
    if not isinstance(parent, vrmlAppearance):
      raise Exception()
    vrmlEntry.__init__(self, parent)
    self.diffuseColor     = [1., 1., 1., 1.]
    self.ambientColor     = [1., 1., 1., 1.]
    self.specularColor    = [0., 0., 0., 1.]
    self.emissiveColor    = [0., 0., 0., 1.]
    self.ambientIntensity = 1.
    self.shininess        = 0.
    self.transparency     = 1.
  def readSpecific(self, fd, string): #FIXME
    #print "%sReading material: %s" % (fillString(self._level), string.replace("\n", "").replace("\t", ""))
    tmp = re.search("transparency\s+([+e\d\.]+)", string, re.I | re.S)
    if tmp != None:
      self.transparency = float(tmp.group(1))
    tmp = re.search("diffuseColor\s+([+e\d\.]+)\s+([+e\d\.]+)\s+([+e\d\.]+)", string, re.I | re.S)
    if tmp != None:
      self.diffuseColor = [float(tmp.group(1)), float(tmp.group(2)), float(tmp.group(3)), 1. - self.transparency]
    tmp = re.search("ambientIntensity\s+([+e\d\.]+)", string, re.I | re.S)
    if tmp != None:
      self.ambientIntensity = float(tmp.group(1))
    tmp = re.search("specularColor\s+([+e\d\.]+)\s+([+e\d\.]+)\s+([+e\d\.]+)", string, re.I | re.S)
    if tmp != None:
      self.specularColor = [float(tmp.group(1)), float(tmp.group(2)), float(tmp.group(3)), 1.]
    tmp = re.search("emissiveColor\s+([+e\d\.]+)\s+([+e\d\.]+)\s+([+e\d\.]+)", string, re.I | re.S)
    if tmp != None:
      self.emissiveColor = [float(tmp.group(1)), float(tmp.group(2)), float(tmp.group(3)), 1.]
    tmp = re.search("shininess\s+([+e\d\.]+)", string, re.I | re.S)
    if tmp != None:
      self.shininess = float(tmp.group(1))
    self.ambientColor = [self.diffuseColor[0] * self.ambientIntensity, 
                         self.diffuseColor[1] * self.ambientIntensity, 
                         self.diffuseColor[2] * self.ambientIntensity, 
                         1. - self.transparency]
    self.diffuseColor[3] = 1. - self.transparency
  def write(self, fd, compList):
    if not self.name in compList:
      ambInt = self.ambientIntensity * 3.
      if ambInt > 1.:
        ambInt = 1.
      fd.write("      appearance Appearance {\n        material DEF %s Material {\n" % self.name)
      fd.write("          diffuseColor %f %f %f\n" % (self.diffuseColor[0], self.diffuseColor[1], self.diffuseColor[2]))
      fd.write("          emissiveColor %f %f %f\n" % (self.emissiveColor[0], self.emissiveColor[1], self.emissiveColor[2]))
      fd.write("          specularColor %f %f %f\n" % (self.specularColor[0], self.specularColor[1], self.specularColor[2]))
      fd.write("          ambientIntensity %f\n" % ambInt)
      fd.write("          transparency %f\n" % self.transparency)
      fd.write("          shininess %f\n" % self.shininess)
      fd.write("        }\n      }\n")
      compList.append(self.name)
    else:
      fd.write("      appearance Appearance {\n        material USE %s\n      }\n" % self.name)

class vrmlTexture(vrmlEntry):
  def __init__(self, parent = None):
    if not isinstance(parent, vrmlAppearance):
      raise Exception()
    vrmlEntry.__init__(self, parent)
    self.texID = None
    self.fname = ""
    self.fpath = ""
  def readSpecific(self, fd, string):
    tmp = re.search("url\s+\"([\w\-\.]+)\"", string, re.I | re.S)
    if tmp != None:
      self.fileName = tmp.group(1)
    self.fpath = os.getcwd()

class mesh:
  def __init__(self):
    self.vertexList  = None
    self.vertexVBO   = 0
    self.normalList  = None
    self.normalVBO   = 0
    self.texList     = None
    self.texVBO      = 0
    self.materials   = []
    self.objects     = []
    self.solid       = False
  def draw(self, rend):
    for mat in self.materials:
      if isinstance(mat, vrmlMaterial):
        rend.setMaterial(mat)
      elif isinstance(mat, vrmlTexture):
        rend.setTexture(mat)
    #glCullFace(GL_BACK)
    if self.solid:
      glEnable(GL_CULL_FACE)
    glEnableClientState(GL_VERTEX_ARRAY)
    glEnableClientState(GL_NORMAL_ARRAY)
    glBindBuffer(GL_ARRAY_BUFFER, self.vertexVBO)
    glVertexPointer(3, GL_FLOAT, 0, None)
    glBindBuffer(GL_ARRAY_BUFFER, self.normalVBO)
    glNormalPointer(GL_FLOAT, 0, None)
    if self.texList != None:
      glEnableClientState(GL_TEXTURE_COORD_ARRAY)
      glBindBuffer(GL_ARRAY_BUFFER, self.texVBO)
      glTexCoordPointer(2, GL_FLOAT, 0, None)
    for obj in self.objects:
      obj.draw()
    #if self.texList != None:
    glDisableClientState(GL_TEXTURE_COORD_ARRAY)
    glDisableClientState(GL_NORMAL_ARRAY)
    glDisableClientState(GL_VERTEX_ARRAY)
    glDisable(GL_CULL_FACE)

class faceset:
  def __init__(self, mode):
    self.mode   = mode
    self.index  = []
    self.length = []
  def append(self, ind, size):
    self.index.append(ind)
    self.length.append(size)
  def draw(self):
    #glEnableClientState(GL_VERTEX_ARRAY)
    #if self.mode != GL_LINES:
      #glEnableClientState(GL_NORMAL_ARRAY)
    #else:
      #glEnable(GL_COLOR_MATERIAL)
      #glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, 0.)
      #glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, [0., 0., 0., 1.])
      #glMaterialfv(GL_FRONT_AND_BACK, GL_EMISSION, [0., 0., 0., 1.])
      #glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
      #glColor4f(0., 0., 1., 1.)
      #glDisable(GL_COLOR_MATERIAL)
    if len(self.index) == 1:
      glDrawArrays(self.mode, self.index[0], self.length[0])
    elif len(self.index) > 1:
      glMultiDrawArrays(self.mode, self.index, self.length, len(self.index))
    #glDisableClientState(GL_NORMAL_ARRAY)
    #glDisableClientState(GL_VERTEX_ARRAY)

class render:
  def __init__(self, aScene):
    self.camera = numpy.matrix([[0.], [20.], [20.], [1.]])
    self.pov    = numpy.matrix([[0.], [0.], [0.], [1.]])
    self.lighta = numpy.matrix([[20.], [20.], [20.], [1.]])
    self.lightb = numpy.matrix([[-20.], [-20.], [-20.], [1.]])
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
    glClearColor(0.0, 0.0, 0.0, 0.0)
    glClearDepth(1.0)
    glDepthFunc(GL_LESS)
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_LIGHTING)
    glEnable(GL_LIGHT1)
    #Setup global lighting
    glLightModelfv(GL_LIGHT_MODEL_AMBIENT, [0.2, 0.2, 0.2, 1.])
    glLightModeli(GL_LIGHT_MODEL_COLOR_CONTROL, GL_SEPARATE_SPECULAR_COLOR)
    #Setup light 0
    glEnable(GL_LIGHT0)
    glLightfv(GL_LIGHT0, GL_POSITION, self.lighta)
    glLightfv(GL_LIGHT0, GL_AMBIENT,  [0.0, 0.0, 0.0, 1.])
    glLightfv(GL_LIGHT0, GL_DIFFUSE,  [1.0, 1.0, 1.0, 1.])
    glLightfv(GL_LIGHT0, GL_SPECULAR, [0.5, 0.5, 0.5, 1.])
    #Setup light 1
    glEnable(GL_LIGHT1)
    glLightfv(GL_LIGHT1, GL_POSITION, self.lightb)
    glLightfv(GL_LIGHT1, GL_AMBIENT,  [0.0, 0.0, 0.0, 1.])
    glLightfv(GL_LIGHT1, GL_DIFFUSE,  [0.6, 0.6, 0.6, 1.])
    glLightfv(GL_LIGHT1, GL_SPECULAR, [0.3, 0.3, 0.3, 1.])
    #glEnable(GL_COLOR_MATERIAL)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    #glEnable(GL_CULL_FACE)
    glCullFace(GL_BACK)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45.0, float(self.width)/float(self.height), 0.1, 1000.0)
    glMatrixMode(GL_MODELVIEW)
  def initScene(self, aScene):
    vrmlShape._vcount = 0
    vrmlShape._pcount = 0
    for entry in aScene.objects:
      if isinstance(entry, vrmlShape) or isinstance(entry, vrmlTransform) or isinstance(entry, vrmlInline):
        self.data.extend(entry.mesh(aScene.transform))
    print "Vertex total count: %d, polygon total count: %d" % (vrmlShape._vcount, vrmlShape._pcount)
    for meshEntry in self.data:
      meshEntry.vertexVBO = glGenBuffers(1)
      #if glIsBuffer(meshEntry.vertexVBO) == GL_FALSE:
        #print "Error creating OpenGL buffer"
        #exit()
      glBindBuffer(GL_ARRAY_BUFFER, meshEntry.vertexVBO)
      glBufferData(GL_ARRAY_BUFFER, meshEntry.vertexList, GL_STATIC_DRAW)
      meshEntry.normalVBO = glGenBuffers(1)
      glBindBuffer(GL_ARRAY_BUFFER, meshEntry.normalVBO)
      glBufferData(GL_ARRAY_BUFFER, meshEntry.normalList, GL_STATIC_DRAW)
      if meshEntry.texList != None:
        meshEntry.texVBO = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, meshEntry.texVBO)
        glBufferData(GL_ARRAY_BUFFER, meshEntry.texList, GL_STATIC_DRAW)
        for mat in meshEntry.materials:
          if isinstance(mat, vrmlTexture):
            self.loadTexture(mat)
  def loadTexture(self, arg):
    im = Image.open(arg.fpath + "/" + arg.fileName)
    try:
      #Get image dimensions and data
      width, height, image = im.size[0], im.size[1], im.tostring("raw", "RGBA", 0, -1)
    except SystemError:
      #Has no alpha channel, synthesize one, see the texture module for more realistic handling
      width, height, image = im.size[0], im.size[1], im.tostring("raw", "RGBX", 0, -1)
    arg.texID = glGenTextures(1)
    #Make it current
    glBindTexture(GL_TEXTURE_2D, arg.texID)
    glPixelStorei(GL_UNPACK_ALIGNMENT,1)
    #Copy the texture into the current texture ID
    #glTexImage2D(GL_TEXTURE_2D, 0, 3, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, image)
    #gluBuild2DMipmaps(GL_TEXTURE_2D, GLU_RGBA8, width, height, GL_RGBA, GL_UNSIGNED_BYTE, image)
    gluBuild2DMipmaps(GL_TEXTURE_2D, 4, width, height, GL_RGBA, GL_UNSIGNED_BYTE, image)
    print "Loaded %s, width: %d, height: %d, id: %d" % (arg.name, width, height, arg.texID)
  def setTexture(self, arg):
    glEnable(GL_TEXTURE_2D)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    #glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
    #glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_DECAL)
    glBindTexture(GL_TEXTURE_2D, arg.texID)
  def setMaterial(self, arg):
    #glDisable(GL_TEXTURE_2D)
    #glDisable(GL_COLOR_MATERIAL)
    glEnable(GL_COLOR_MATERIAL)
#    glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, arg.diffuseColor)
    glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, arg.shininess * 128.)
    glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, arg.specularColor)
    glMaterialfv(GL_FRONT_AND_BACK, GL_EMISSION, arg.emissiveColor)
    #glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT, arg.ambientColor)
    #glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, arg.diffuseColor)
    #glColorMaterial(GL_FRONT_AND_BACK, GL_EMISSION)
    #glColor3f(arg.emissiveColor[0], arg.emissiveColor[1], arg.emissiveColor[2])
    #glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT)
    glColor4f(arg.ambientColor[0], arg.ambientColor[1], arg.ambientColor[2], arg.ambientColor[3])
    glColorMaterial(GL_FRONT_AND_BACK, GL_DIFFUSE)
    glColor4f(arg.diffuseColor[0], arg.diffuseColor[1], arg.diffuseColor[2], arg.diffuseColor[3])
    #glColorMaterial(GL_FRONT_AND_BACK, GL_SPECULAR)
    #glColor3f(arg.specularColor[0], arg.specularColor[1], arg.specularColor[2])
    #glEnable(GL_COLOR_MATERIAL)
    glDisable(GL_COLOR_MATERIAL)
  def drawAxis(self):
    glDisable(GL_TEXTURE_2D)
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
    if self.updated == True:
      self.updated = False
      glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
      glLoadIdentity()
      gluLookAt(float(self.camera[0]), float(self.camera[1]), float(self.camera[2]), 
                float(self.pov[0]), float(self.pov[1]), float(self.pov[2]), 
                float(self.axis[0]), float(self.axis[1]), float(self.axis[2]))
      glLightfv(GL_LIGHT0, GL_POSITION, self.lighta)
      glLightfv(GL_LIGHT1, GL_POSITION, self.lightb)
      self.drawAxis()
      for current in self.data:
        current.draw(self)
      glutSwapBuffers()
      self.fps += 1
      if time.time() - self.cntr >= 1.:
        glutSetWindowTitle("VRML viewer: %d vertices, %d polygons, %d FPS" % (vrmlShape._vcount, vrmlShape._pcount, self.fps / (time.time() - self.cntr)))
        #print "FPS: %d" % (self.fps / (time.time() - self.cntr))
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
    if bNumber == GLUT_MIDDLE_BUTTON:
      if bAction == GLUT_DOWN:
        self.moveCamera = True
        self.mousePos = [xPos, yPos]
      else:
        self.moveCamera = False
    if bNumber == 3 and bAction == GLUT_DOWN:
      zm = 0.9
      scaleMatrix = numpy.matrix([[zm, 0., 0., 0.],
                                  [0., zm, 0., 0.],
                                  [0., 0., zm, 0.],
                                  [0., 0., 0., 1.]])
      self.camera -= self.pov
      self.camera = scaleMatrix * self.camera
      self.camera += self.pov
    if bNumber == 4 and bAction == GLUT_DOWN:
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
    if self.rotateCamera == True:
      self.camera -= self.pov
      zrot = (self.mousePos[0] - xPos) / 100.
      nrot = (yPos - self.mousePos[1]) / 100.
      if zrot != 0.:
        #if self.axis[2] < 0:
          #zrot = -zrot;
        rotMatrixA = numpy.matrix([[math.cos(zrot), -math.sin(zrot), 0., 0.],
                                   [math.sin(zrot),  math.cos(zrot), 0., 0.],
                                   [            0.,              0., 1., 0.],
                                   [            0.,              0., 0., 1.]])
        self.camera = rotMatrixA * self.camera
      if nrot != 0.:
        #normal = getNormal(self.camera, [0., 0., 1.])
        normal = getNormal(self.camera, self.axis)
        normal /= numpy.linalg.norm(normal)
        #angle = getAngle(self.camera, [0., 0., 1.])
        angle = getAngle(self.camera, self.axis)
        if (nrot > 0 and nrot > angle) or (nrot < 0 and -nrot > math.pi - angle):
          self.axis = -self.axis
        rotMatrixB = fillRotateMatrix(normal, nrot)
        self.camera = rotMatrixB * self.camera
      self.camera += self.pov
      self.mousePos = [xPos, yPos]
    elif self.moveCamera == True:
      tlVector = numpy.matrix([[(xPos - self.mousePos[0]) / 50.], [(self.mousePos[1] - yPos) / 50.], [0.], [0.]])
      self.camera -= self.pov
      normal = getNormal([0., 0., 1.], self.camera)
      normal /= numpy.linalg.norm(normal)
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
    if key == "\x1b" or key == "q" or key == "Q":
      exit()
    if key == "r" or key == "R":
      self.camera = numpy.matrix([[0.], [20.], [0.], [1.]])
      self.pov    = numpy.matrix([[0.], [0.], [0.], [1.]])
    self.updated = True

parser = optparse.OptionParser()
parser.add_option("-v", "--view", dest="view", help="Render and show model.", default=False, action="store_true")
parser.add_option("-w", "--write", dest="rebuild", help="Rebuild model.", default=False, action="store_true")
parser.add_option("-t", "--translate", dest="translate", help="Move shapes to new coordinates [x,y,z], default value \"0.,0.,0.\".", default='0.,0.,0.')
parser.add_option("-r", "--rotate", dest="rotate", help="Rotate shapes around vector [x,y,z] by angle in degrees, default value \"0.,0.,1.,0.\".", default='0.,0.,1.,0.')
parser.add_option("-s", "--scale", dest="scale", help="Scale shapes by [x,y,z], default value \"1.,1.,1.\".", default='1.,1.,1.')
(options, args) = parser.parse_args()

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

gRotate[3] *= (180 / math.pi)
sc = vrmlScene()
sc.setTransform(gTranslate, gRotate, gScale)

for fileName in args:
  if os.path.isfile(fileName):
    sc.loadFile(fileName)
    if options.rebuild == True:
      sc.saveFile(os.path.splitext(fileName)[0] + ".re.wrl")

def hprint(obj, level = 0):
  for i in obj.objects:
    print "%s%s - %s" % (fillString(level), i.__class__.__name__, i.name)
    hprint(i, level + 2)

print "----------------STRUCTURE---------------"
hprint(sc)
print "----------------END STRUCTURE-----------"

if options.view == True:
  rend = render(sc)
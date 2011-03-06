#!/usr/bin/env python
# -*- coding: utf-8 -*-
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

def getChunk(fd):
  balance = 1
  content = ""
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
        fd.seek(-(len(data) - i - 2), os.SEEK_CUR)
        return content + data[0:i - 1]
    content += data
  return ""

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

def calcBalance(string):
  balance = 0
  offset = 0
  for i in range(0, len(string)):
    if string[i] == '{' or string[i] == '[':
      balance += 1
    if string[i] == '}' or string[i] == ']':
      balance -= 1
      offset = len(string) - i - 2
  return (balance, offset)

class vrmlScene:
  def __init__(self):
    self.objects = []
    self.entries = []
    self.transform = numpy.matrix([[1., 0., 0., 0.],
                                   [0., 1., 0., 0.],
                                   [0., 0., 1., 0.],
                                   [0., 0., 0., 1.]])
  def setTransform(self, aTrans, aRot, aScale):
    translation = numpy.matrix([[1., 0., 0., aTrans[0]],
                                [0., 1., 0., aTrans[1]],
                                [0., 0., 1., aTrans[2]],
                                [0., 0., 0., 1.]])
    rotation = fillRotateMatrix(aRot, aRot[3]);
    scale = numpy.matrix([[aScale[0], 0., 0., 0.],
                          [0., aScale[1], 0., 0.],
                          [0., 0., aScale[2], 0.],
                          [0., 0., 0., 1.]])
    self.transform = translation * rotation * scale
  def loadFile(self, fileName):
    wrlFile = open(fileName, "r")
    oldDir = os.getcwd()
    os.chdir(os.path.dirname(fileName))
    defPattern = re.compile("([\w\-]*?)\s*(\w+)\s*{", re.I | re.S)
    line = 0
    while 1:
      wrlContent = wrlFile.readline()
      if len(wrlContent) == 0:
        break
      tmp = defPattern.search(wrlContent)
      if tmp != None:
        print "\nEntry: '%s' '%s'" % (tmp.group(1), tmp.group(2))
        subname = tmp.group(1)
        if tmp.group(2) == "Transform":
          entry = vrmlObject(self)
          entry.name = tmp.group(1)
          entry.read(wrlFile)
          self.objects.append(entry)
        elif tmp.group(2) == "Shape":
          newShape = vrmlShape(self)
          newShape.name = subname
          newShape.read(wrlFile)
          self.objects.append(newShape)
        else:
          skipChunk(wrlFile)
    os.chdir(oldDir)
    wrlFile.close()
  def saveFile(self, fileName):
    wrlFile = open(fileName, "w")
    wrlFile.write("#VRML V2.0 utf8\n#Exported from Blender by wrlconv.py\n")
    for entry in self.objects:
      entry.write(wrlFile, self.transform)
    wrlFile.close()

class vrmlObject:
  def __init__(self, parent):
    self.parent = parent
    self.name = ""
    self.subobject = []
    self.transform = numpy.matrix([[1., 0., 0., 0.],
                                   [0., 1., 0., 0.],
                                   [0., 0., 1., 0.],
                                   [0., 0., 0., 1.]])
  def read(self, fd):
    print "Read object start"
    defPattern = re.compile("([\w\-]*?)\s*(\w+)\s*{", re.I | re.S)
    delta, offset, balance = 0, 0, 0
    #Highest level
    while 1:
      data = fd.readline()
      if len(data) == 0:
        break
      (delta, offset) = calcBalance(data)
      balance += delta
      if balance < 0:
        print "  Wrong highbalance: %d" % balance
        fd.seek(-offset, os.SEEK_CUR)
        break
      high = defPattern.search(data)
      if high != None:
        lowbalance = balance
        print "  vrmlObject: '%s' '%s'" % (high.group(1), high.group(2))
        if high.group(2) == "Transform":
          newObj = vrmlObject(self)
          newObj.name = high.group(1)
          newObj.read(fd)
          self.subobject.append(newObj)
          balance -= delta
        elif high.group(2) == "Group":
          subname = high.group(1)
          #Lowest level
          while 1:
            subdata = fd.readline()
            if len(subdata) == 0:
              break
            (delta, offset) = calcBalance(subdata)
            balance += delta
            if balance < lowbalance:
              print "    Wrong lowbalance: %d" % balance
              fd.seek(-offset, os.SEEK_CUR)
              break
            low = defPattern.search(subdata)
            if low != None:
              print "    SubGroup: '%s' '%s'" % (low.group(1), low.group(2))
              if low.group(2) == "Shape":
                newShape = vrmlShape(self)
                newShape.name = subname
                newShape.read(fd)
                ptr = self
                while not isinstance(ptr, vrmlScene):
                  ptr = ptr.parent
                self.subobject.append(newShape)
                ptr.entries.append(newShape)
              else:
                skipChunk(fd)
              balance -= delta
        elif high.group(2) == "Shape":
          newShape = vrmlShape(self)
          newShape.read(fd)
          ptr = self
          while not isinstance(ptr, vrmlScene):
            ptr = ptr.parent
          self.subobject.append(newShape)
          ptr.entries.append(newShape)
          balance -= delta
        else:
          skipChunk(fd)
          balance -= delta
      else:
        translation = None
        rotation = None
        scale = None
        tmp = re.search("translation\s+([+e\d\.\-]+)\s+([+e\d\.\-]+)\s+([+e\d\.\-]+)", data, re.I | re.S)
        if tmp != None:
          translation = numpy.matrix([[1., 0., 0., float(tmp.group(1))],
                                      [0., 1., 0., float(tmp.group(2))],
                                      [0., 0., 1., float(tmp.group(3))],
                                      [0., 0., 0., 1.]])
        tmp = re.search("rotation\s+([+e\d\.\-]+)\s+([+e\d\.\-]+)\s+([+e\d\.\-]+)\s+([+e\d\.\-]+)", data, re.I | re.S)
        if tmp != None:
          rotation = fillRotateMatrix([float(tmp.group(1)), float(tmp.group(2)), float(tmp.group(3))], float(tmp.group(4)))
        tmp = re.search("scale\s+([+e\d\.\-]+)\s+([+e\d\.\-]+)\s+([+e\d\.\-]+)", data, re.I | re.S)
        if tmp != None:
          scale = numpy.matrix([[float(tmp.group(1)), 0., 0., 0.],
                                [0., float(tmp.group(2)), 0., 0.],
                                [0., 0., float(tmp.group(3)), 0.],
                                [0., 0., 0., 1.]])
        tmp = re.search("USE\s+([\w\-]+)", data, re.I | re.S)
        if tmp != None:
          ptr = self
          while not isinstance(ptr, vrmlScene):
            ptr = ptr.parent
          for sh in ptr.entries:
            if sh.name == tmp.group(1):
              print "  Found shape %s" % sh.name
              self.subobject.append(sh)
        if translation != None:
          self.transform = self.transform * translation
        if rotation != None:
          self.transform = self.transform * rotation
        if scale != None:
          self.transform = self.transform * scale
    print "Read object end\n"
  def getMesh(self, transform):
    res = []
    tform = transform * self.transform
    for obj in self.subobject:
      if isinstance(obj, vrmlShape) or isinstance(obj, vrmlObject):
        res.extend(obj.getMesh(tform))
    return res
  def write(self, fd, transform):
    tform = transform * self.transform
    for obj in self.subobject:
      if isinstance(obj, vrmlShape) or isinstance(obj, vrmlObject):
        obj.write(fd, tform)

class vrmlShape:
  def __init__(self, parent):
    self.parent = parent
    self.name = ""
    self.subobject  = []
  def read(self, fd):
    defPattern = re.compile("([\w\-]*?)\s*(\w+)\s*{", re.I | re.S)
    delta, offset, balance = 0, 0, 0
    #Highest level
    while 1:
      data = fd.readline()
      if len(data) == 0:
        break
      (delta, offset) = calcBalance(data)
      balance += delta
      if balance < 0:
        print "    Wrong highbalance: %d" % balance
        fd.seek(-offset, os.SEEK_CUR)
        break
      high = defPattern.search(data)
      if high != None:
        print "    vrmlShape: '%s' '%s', balance: %d" % (high.group(1), high.group(2), balance)
        midbalance = balance
        if high.group(2) == "Appearance":
          #Middle level
          while 1:
            subdata = fd.readline()
            if len(subdata) == 0:
              break
            (delta, offset) = calcBalance(subdata)
            balance += delta
            if balance < midbalance:
              print "      Wrong balance: %d" % balance
              fd.seek(-offset, os.SEEK_CUR)
              break
            middle = defPattern.search(subdata)
            if middle != None:
              print "      SubAppearance: '%s' '%s', balance: %d" % (middle.group(1), middle.group(2), balance)
              if middle.group(2) == "Material":
                print "      Material: %s" % middle.group(1)
                matData = getChunk(fd)
                newMat = vrmlMaterial(self)
                newMat.name = middle.group(1)
                newMat.read(matData)
                self.subobject.append(newMat)
                ptr = self
                while not isinstance(ptr, vrmlScene):
                  ptr = ptr.parent
                ptr.entries.append(newMat)
              elif middle.group(2) == "ImageTexture":
                print "      ImageTexture: %s" % middle.group(1)
                texData = getChunk(fd)
                newTex = vrmlTexture(self)
                newTex.name = middle.group(1)
                newTex.read(texData)
                self.subobject.append(newTex)
                ptr = self
                while not isinstance(ptr, vrmlScene):
                  ptr = ptr.parent
                ptr.entries.append(newTex)
              else:
                skipChunk(fd)
              balance -= delta
            else:
              tmp = re.search("material\s+USE\s+([\w\-]+)", subdata, re.I | re.S)
              if tmp != None:
                ptr = self
                while isinstance(ptr, vrmlScene) == False:
                  ptr = ptr.parent
                for mat in ptr.entries:
                  if mat.name == tmp.group(1):
                    print "      Found mat %s" % mat.name
                    linkedMaterial = copy.copy(mat)
                    linkedMaterial.linked = True
                    self.subobject.append(linkedMaterial)
        elif high.group(2) == "IndexedFaceSet":
          newGeo = vrmlGeometry(self)
          newGeo.read(fd)
          self.subobject.append(newGeo)
          ptr = self
          while not isinstance(ptr, vrmlScene):
            ptr = ptr.parent
          ptr.entries.append(newGeo)
          balance -= delta
        else:
          skipChunk(fd)
          balance -= delta
    print "      End shape read"
  def getMesh(self, transform):
    print "Draw object %s" % self.name
    newMesh = mesh()
    for obj in self.subobject:
      if isinstance(obj, vrmlMaterial) or isinstance(obj, vrmlTexture):
        newMesh.materials.append(obj)
      elif isinstance(obj, vrmlGeometry):
        print "  Draw subobject %s" % obj.name
        if len(obj.polygons) == len(obj.polygonsUV):
          genTex = True
        else:
          genTex = False
        tsa = time.time()
        newMesh.smooth = obj.smooth
        tmpVertices = []
        for vert in obj.vertices:
          tmp = numpy.matrix([[vert[0]], [vert[1]], [vert[2]], [1.]])
          tmp = transform * tmp
          tmpVertices.append(numpy.array([float(tmp[0]), float(tmp[1]), float(tmp[2])]))
        newMesh.arraySizes.append(obj.triCount)
        newMesh.arraySizes.append(obj.quadCount)
        newMesh.arraySizes.append(obj.polyCount)
        length = (obj.triCount + obj.quadCount + obj.polyCount)
        newMesh.vertexList = numpy.zeros(length * 3, dtype = numpy.float32)
        newMesh.normalList = numpy.zeros(length * 3, dtype = numpy.float32)
        if genTex == True:
          newMesh.texList = numpy.zeros(length * 2, dtype = numpy.float32)
        tPos = 0
        qPos = obj.triCount
        pPos = obj.triCount + obj.quadCount
        if newMesh.smooth == False:
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
              pos = pPos
              pPos += len(obj.polygons[poly])
            for ind in range(0, len(obj.polygons[poly])):
              newMesh.vertexList[3 * pos]     = tmpVertices[obj.polygons[poly][ind]][0]
              newMesh.normalList[3 * pos]     = normal[0]
              newMesh.vertexList[3 * pos + 1] = tmpVertices[obj.polygons[poly][ind]][1]
              newMesh.normalList[3 * pos + 1] = normal[1]
              newMesh.vertexList[3 * pos + 2] = tmpVertices[obj.polygons[poly][ind]][2]
              newMesh.normalList[3 * pos + 2] = normal[2]
              if genTex == True:
                newMesh.texList[2 * pos]     = obj.verticesUV[obj.polygonsUV[poly][ind]][0]
                newMesh.texList[2 * pos + 1] = obj.verticesUV[obj.polygonsUV[poly][ind]][1]
              pos += 1
        else:
          tmpNormals = []
          for i in range(0, len(obj.vertices)):
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
          for poly in range(0, len(obj.polygons)):
            pos = 0
            if len(obj.polygons[poly]) == 3:
              pos = tPos
              tPos += 3
            elif len(obj.polygons[poly]) == 4:
              pos = qPos
              qPos += 4
            else:
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
                newMesh.texList[2 * pos]     = obj.verticesUV[obj.polygonsUV[poly][ind]][0]
                newMesh.texList[2 * pos + 1] = obj.verticesUV[obj.polygonsUV[poly][ind]][1]
              pos += 1
        tsb = time.time()
        print "Created in: %f" % (tsb - tsa)
    return [newMesh]
  def write(self, fd, transform):
    print "Write object %s" % self.name
    fd.write("DEF %s Transform {\n  children [\n" % self.name)
    fd.write("    Shape {\n")
    for obj in self.subobject:
      if isinstance(obj, vrmlMaterial):
        obj.write(fd)
      elif isinstance(obj, vrmlGeometry):
        fd.write("      geometry IndexedFaceSet {\n        coord Coordinate { point [\n")
        for i in range(0, len(obj.vertices)):
          tmp = numpy.matrix([[obj.vertices[i][0]], [obj.vertices[i][1]], [obj.vertices[i][2]], [1.]])
          tmp = transform * tmp
          fd.write("          %f %f %f" % (float(tmp[0]), float(tmp[1]), float(tmp[2])))
          if i != len(obj.vertices) - 1:
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

class vrmlGeometry:
  def __init__(self, parent):
    self.parent = parent
    self.name = ""
    self.smooth = False
    self.vertices   = []
    self.polygons   = []
    self.verticesUV = []
    self.polygonsUV = []
    self.triCount   = 0
    self.quadCount  = 0
    self.polyCount  = 0
  def read(self, fd):
    defPattern = re.compile("([\w\-]*?)\s*(\w+)\s*{", re.I | re.S)
    dataPattern = re.compile("(\w+)\s*\[", re.I | re.S)
    delta, offset, balance = 0, 0, 0
    #Highest level
    while 1:
      data = fd.readline()
      if len(data) == 0:
        break
      (delta, offset) = calcBalance(data)
      balance += delta
      if balance < 0:
        print "        Wrong highbalance: %d" % balance
        fd.seek(-offset, os.SEEK_CUR)
        break
      high = defPattern.search(data)
      if high != None:
        lowbalance = balance
        print "        SubSet: '%s' '%s', balance: %d delta: %d" % (high.group(1), high.group(2), balance, delta)
        if high.group(2) == "Coordinate":
          print "        Rename: %s -> %s" % (self.name, high.group(1))
          self.name = high.group(1)
          self.readVertices(fd)
        elif high.group(2) == "TextureCoordinate":
          self.readVerticesUV(fd)
        else:
          skipChunk(fd)
        balance -= delta
      else:
        high = dataPattern.search(data)
        if high != None:
          print "      SubSet: '%s', balance: %d" % (high.group(1), balance)
          if high.group(1) == "coordIndex":
            self.readPolygons(fd)
          elif high.group(1) == "texCoordIndex":
            self.readPolygonsUV(fd)
          else:
            skipChunk(fd)
          balance -= delta
        else:
          high = re.search("solid\s+(TRUE|FALSE)", data, re.I | re.S) #FIXME
          if high != None:
            if high.group(1) == "TRUE":
              self.smooth = False
            else:
              self.smooth = True
          high = re.search("coord\s+USE\s+([\w\-]+)", data, re.I | re.S)
          if high != None:
            ptr = self
            while isinstance(ptr, vrmlScene) == False:
              ptr = ptr.parent
            for geo in ptr.entries:
              if geo.name == high.group(1):
                print "        Found geometry %s" % geo.name
                self.vertices   = geo.vertices
    print "      End geometry read"
  def readVertices(self, fd):
    print "      Start vertex read"
    vertexPattern = re.compile("([+e\d\-\.]+)[ ,\t]+([+e\d\-\.]+)[ ,\t]+([+e\d\-\.]+)", re.I | re.S)
    delta, offset, balance = 0, 0, 0
    while 1:
      data = fd.readline()
      if len(data) == 0:
        break
      (delta, offset) = calcBalance(data)
      balance += delta
      high = vertexPattern.search(data)
      if high != None:
        self.vertices.append(numpy.array([float(high.group(1)), float(high.group(2)), float(high.group(3))]))
      if balance < 0:
        print "      Wrong balance: %d" % balance
        fd.seek(-offset, os.SEEK_CUR)
        break
    print "      End vertex read, loaded %d" % len(self.vertices)
  def readVerticesUV(self, fd):
    print "      Start UV vertex read"
    vertexPattern = re.compile("([+e\d\-\.]+)[ ,\t]+([+e\d\-\.]+)", re.I | re.S)
    delta, offset, balance = 0, 0, 0
    while 1:
      data = fd.readline()
      if len(data) == 0:
        break
      (delta, offset) = calcBalance(data)
      balance += delta
      vPos = 0
      while 1:
        high = vertexPattern.search(data, vPos)
        if high != None:
          self.verticesUV.append(numpy.array([float(high.group(1)), float(high.group(2))]))
          vPos = high.end()
        else:
          break
      if balance < 0:
        print "      Wrong balance: %d" % balance
        fd.seek(-offset, os.SEEK_CUR)
        break
    print "      End UV vertex read, loaded %d" % len(self.verticesUV)
  def readPolygonsUV(self, fd):
    print "      Start UV polygon read"
    polyPattern = re.compile("([ ,\t\d]+)-1", re.I | re.S)
    indPattern = re.compile("[ ,\t]*(\d+)[ ,\t]*", re.I | re.S)
    delta, offset, balance = 0, 0, 0
    while 1:
      data = fd.readline()
      if len(data) == 0:
        break
      (delta, offset) = calcBalance(data)
      balance += delta
      high = polyPattern.search(data)
      if high != None:
        polyData = []
        indPos = 0
        while 1:
          ind = indPattern.search(high.group(1), indPos)
          if ind == None:
            break
          polyData.append(int(ind.group(1)))
          indPos = ind.end()
        self.polygonsUV.append(polyData)
      if balance < 0:
        print "      Wrong balance: %d" % balance
        fd.seek(-offset, os.SEEK_CUR)
        break
    print "      End UV polygon read, loaded poly: %d" % len(self.polygons)
  def readPolygons(self, fd):
    print "      Start polygon read"
    polyPattern = re.compile("([ ,\t\d]+)-1", re.I | re.S)
    indPattern = re.compile("[ ,\t]*(\d+)[ ,\t]*", re.I | re.S)
    delta, offset, balance = 0, 0, 0
    while 1:
      data = fd.readline()
      if len(data) == 0:
        break
      (delta, offset) = calcBalance(data)
      balance += delta
      high = polyPattern.search(data)
      if high != None:
        polyData = []
        indPos = 0
        while 1:
          ind = indPattern.search(high.group(1), indPos)
          if ind == None:
            break
          polyData.append(int(ind.group(1)))
          indPos = ind.end()
        if len(polyData) == 3:
          self.triCount += 3
        elif len(polyData) == 4:
          self.quadCount += 4
        else:
          self.polyCount += len(polyData)
        self.polygons.append(polyData)
      if balance < 0:
        print "      Wrong balance: %d" % balance
        fd.seek(-offset, os.SEEK_CUR)
        break
    print "      End polygon read, loaded tri: %d, quad: %d, poly: %d, vTOT: %d" % (self.triCount / 3, self.quadCount / 4, len(self.polygons) - self.triCount / 3 - self.quadCount / 4, self.polyCount + self.triCount + self.quadCount)

class vrmlMaterial:
  def __init__(self, parent):
    self.parent = parent
    self.name = ""
    self.linked = False
    self.diffuseColor     = [0., 0., 0., 1.]
    self.ambientColor     = [0., 0., 0., 1.]
    self.specularColor    = [0., 0., 0., 1.]
    self.emissiveColor    = [0., 0., 0., 1.]
    self.ambientIntensity = 0.
    self.shininess        = 0.
    self.transparency     = 0.
  def read(self, fileContent):
    tmp = re.search("transparency\s+([+e\d\.]+)", fileContent, re.I | re.S)
    if tmp != None:
      self.transparency = float(tmp.group(1))
    tmp = re.search("diffuseColor\s+([+e\d\.]+)\s+([+e\d\.]+)\s+([+e\d\.]+)", fileContent, re.I | re.S)
    if tmp != None:
      self.diffuseColor = [float(tmp.group(1)), float(tmp.group(2)), float(tmp.group(3)), 1. - self.transparency]
    tmp = re.search("ambientIntensity\s+([+e\d\.]+)", fileContent, re.I | re.S)
    if tmp != None:
      self.ambientIntensity = float(tmp.group(1))
      self.ambientColor = [self.diffuseColor[0] * self.ambientIntensity, 
                           self.diffuseColor[1] * self.ambientIntensity, 
                           self.diffuseColor[2] * self.ambientIntensity, 
                           1. - self.transparency]
    tmp = re.search("specularColor\s+([+e\d\.]+)\s+([+e\d\.]+)\s+([+e\d\.]+)", fileContent, re.I | re.S)
    if tmp != None:
      self.specularColor = [float(tmp.group(1)), float(tmp.group(2)), float(tmp.group(3)), 1.]
    tmp = re.search("emissiveColor\s+([+e\d\.]+)\s+([+e\d\.]+)\s+([+e\d\.]+)", fileContent, re.I | re.S)
    if tmp != None:
      self.emissiveColor = [float(tmp.group(1)), float(tmp.group(2)), float(tmp.group(3)), 1.]
    tmp = re.search("shininess\s+([+e\d\.]+)", fileContent, re.I | re.S)
    if tmp != None:
      self.shininess = float(tmp.group(1))
  def write(self, fd):
    if self.linked == False:
      aint = self.ambientIntensity * 3.
      if aint > 1.:
        aint = 1.
      fd.write("      appearance Appearance {\n        material DEF %s Material {\n" % self.name)
      fd.write("          diffuseColor %f %f %f\n" % (self.diffuseColor[0], self.diffuseColor[1], self.diffuseColor[2]))
      fd.write("          emissiveColor %f %f %f\n" % (self.emissiveColor[0], self.emissiveColor[1], self.emissiveColor[2]))
      fd.write("          specularColor %f %f %f\n" % (self.specularColor[0], self.specularColor[1], self.specularColor[2]))
      fd.write("          ambientIntensity %f\n" % aint)
      fd.write("          transparency %f\n" % self.transparency)
      fd.write("          shininess %f\n" % self.shininess)
      fd.write("        }\n      }\n")
    else:
      fd.write("      appearance Appearance {\n        material USE %s\n      }\n" % self.name)

class vrmlTexture:
  def __init__(self, parent):
    self.parent = parent
    self.texID = None
    self.fname = ""
    self.fpath = ""
    self.name = ""
  def read(self, fileContent):
    tmp = re.search("url\s+\"([\w\-\.]+)\"", fileContent, re.I | re.S)
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
    self.smooth      = False
    self.arraySizes  = []

class render:
  def __init__(self, aScene):
    self.camera = numpy.matrix([[0.], [20.], [20.], [1.]])
    self.pov    = numpy.matrix([[0.], [0.], [0.], [1.]])
    self.light  = numpy.matrix([[20.], [20.], [20.], [1.]])
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
    glEnable(GL_LIGHT0)
    glLightfv(GL_LIGHT0, GL_POSITION, self.light)
    #glEnable(GL_COLOR_MATERIAL)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glEnable(GL_CULL_FACE)
    glCullFace(GL_BACK)
    #glShadeModel(GL_FLAT)
    #glLightModel(GL_LIGHT_MODEL_COLOR_CONTROL, GL_SEPARATE_SPECULAR_COLOR)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45.0, float(self.width)/float(self.height), 0.1, 1000.0)
    glMatrixMode(GL_MODELVIEW)
  def initScene(self, aScene):
    for entry in aScene.objects:
      self.data.extend(entry.getMesh(aScene.transform))
    for meshEntry in self.data:
      print "Vertex count: %d" % len(meshEntry.vertexList)
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
    print "        Loaded %s, width: %d, height: %d, id: %d" % (arg.name, width, height, arg.texID)
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
                float(self.pov[0]), float(self.pov[1]), float(self.pov[2]), 0., 0., 1.)
      glLightfv(GL_LIGHT0, GL_POSITION, self.light)
      self.drawAxis()
      glEnableClientState(GL_VERTEX_ARRAY)
      glEnableClientState(GL_NORMAL_ARRAY)
      for current in self.data:
        for mat in current.materials:
          if isinstance(mat, vrmlMaterial):
            self.setMaterial(mat)
            #print "Set material"
          elif isinstance(mat, vrmlTexture):
            self.setTexture(mat)
            #print "Set texture"
        if current.texList != None:
          #print "Set Tex Coords"
          glEnableClientState(GL_TEXTURE_COORD_ARRAY)
          glBindBuffer(GL_ARRAY_BUFFER, current.texVBO)
          glTexCoordPointer(2, GL_FLOAT, 0, None)
        glBindBuffer(GL_ARRAY_BUFFER, current.vertexVBO)
        glVertexPointer(3, GL_FLOAT, 0, None)
        glBindBuffer(GL_ARRAY_BUFFER, current.normalVBO)
        glNormalPointer(GL_FLOAT, 0, None)
        glDrawArrays(GL_TRIANGLES, 0, current.arraySizes[0])
        glDrawArrays(GL_QUADS, current.arraySizes[0], current.arraySizes[1])
        glDrawArrays(GL_POLYGON, current.arraySizes[0] + current.arraySizes[1], current.arraySizes[2])
        if current.texList != None:
          glDisableClientState(GL_TEXTURE_COORD_ARRAY)
      glDisableClientState(GL_NORMAL_ARRAY)
      glDisableClientState(GL_VERTEX_ARRAY)
      glutSwapBuffers()
      self.fps += 1
      if time.time() - self.cntr >= 1.:
        print "FPS: %d" % (self.fps / (time.time() - self.cntr))
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
      normal = getNormal(self.camera, [0., 0., 1.])
      normal /= numpy.linalg.norm(normal)
      zrot = (self.mousePos[0] - xPos) / 100.
      nrot = (yPos - self.mousePos[1]) / 100.
      if zrot != 0.:
        rotMatrixA = numpy.matrix([[math.cos(zrot), -math.sin(zrot), 0., 0.],
                                   [math.sin(zrot),  math.cos(zrot), 0., 0.],
                                   [            0.,              0., 1., 0.],
                                   [            0.,              0., 0., 1.]])
        self.camera = rotMatrixA * self.camera
      if nrot != 0.:
        angle = getAngle(self.camera, [0., 0., 1.])
        if not ((nrot > 0 and nrot > angle) or (nrot < 0 and -nrot > math.pi - angle)):
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

for fname in args:
  sc.loadFile(fname)
  if options.rebuild == True:
    sc.saveFile(fname + "~")

if options.view == True:
  rend = render(sc)
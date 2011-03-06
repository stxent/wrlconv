#!/usr/bin/env python
# -*- coding: utf-8 -*-
import re
import math
import numpy
from optparse import OptionParser
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
import sys
import time

from OpenGL.arrays import ArrayDatatype as ADT
from OpenGL.GL import *
from OpenGL.raw import GL

#class vertexBuffer(object):
  #def __init__(self, data, usage):
    #self.buffer = GL.GLuint(0)
    #glGenBuffers(1, self.buffer)
    #self.buffer = self.buffer.value
    #glBindBuffer(GL_ARRAY_BUFFER, self.buffer)
    #glBufferData(GL_ARRAY_BUFFER, ADT.arrayByteCount(data), ADT.voidDataPointer(data), usage)

  #def __del__(self):
    #glDeleteBuffers(1, GL.GLuint(self.buffer))

  #def bind(self):
    #glBindBuffer(GL_ARRAY_BUFFER, self.buffer)

  #def bind_colors(self, size, type, stride=0):
    #self.bind()
    #glColorPointer(size, type, stride, None)

  #def bind_edgeflags(self, stride=0):
    #self.bind()
    #glEdgeFlagPointer(stride, None)

  #def bind_indexes(self, type, stride=0):
    #self.bind()
    #glIndexPointer(type, stride, None)

  #def bind_normals(self, type, stride=0):
    #self.bind()
    #glNormalPointer(type, stride, None)

  #def bind_texcoords(self, size, type, stride=0):
    #self.bind()
    #glTexCoordPointer(size, type, stride, None)

  #def bind_vertexes(self, size, type, stride=0):
    #self.bind()
    #glVertexPointer(size, type, stride, None)


def fillRotateMatrix(v, angle):
  cs = math.cos(angle)
  sn = math.sin(angle)
  v = [float(v[0]), float(v[1]), float(v[2])]
  return numpy.matrix([[     cs + v[0]*v[0]*(1 - cs), v[0]*v[1]*(1 - cs) - v[2]*sn, v[0]*v[2]*(1 - cs) + v[1]*sn, 0.],
                       [v[1]*v[0]*(1 - cs) + v[2]*sn,      cs + v[1]*v[1]*(1 - cs), v[1]*v[2]*(1 - cs) - v[0]*sn, 0.],
                       [v[2]*v[0]*(1 - cs) - v[1]*sn, v[2]*v[1]*(1 - cs) + v[0]*sn,      cs + v[2]*v[2]*(1 - cs), 0.],
                       [                          0.,                           0.,                           0., 1.]])

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

def getChunkEnd(arg, start):
  balance = 1
  for i in range(start, len(arg)):
    if arg[i] == '{' or arg[i] == '[':
      balance += 1
    if arg[i] == '}' or arg[i] == ']':
      balance -= 1
    if balance == 0:
      return (i - 1)
  return None

class scene:
  def __init__(self):
    self.materials = []
    self.entities = []
    self.shapes = []
    self.transform = numpy.matrix([[1., 0., 0., 0.],
                                   [0., 1., 0., 0.],
                                   [0., 0., 1., 0.],
                                   [0., 0., 0., 1.]])
  def clear(self):
    self.materials = []
    self.entities = []
    self.shapes = []
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
    wrlContent = wrlFile.read()
    wrlFile.close()
    contentPos = 0
    transformPattern = re.compile("DEF\s+(\w+)\s+Transform\s+{", re.I | re.S)
    vertexPattern = re.compile("coord[\s\w]+{\s+point\s+\[([+e,\s\d\.\-]*)\]\s+}", re.I | re.S)
    indexPattern = re.compile("coordIndex[\s]+\[([,\s\d\-]*)\]", re.I | re.S)
    while (1):
      tmp = transformPattern.search(wrlContent, contentPos)
      if tmp == None:
        break
      startPos = tmp.end()
      endPos = getChunkEnd(wrlContent, tmp.end())
      if endPos == None:
        break
      newEntity = entity()
      newEntity.name = tmp.group(1)
      print 'Object %s at %d' % (newEntity.name, tmp.start())
      newEntity.findTransform(wrlContent[startPos:endPos])
      newEntity.transform = newEntity.transform * self.transform
      existing = re.search("children\s+\[\s+USE\s+(\w+)\s+\]", wrlContent[startPos:endPos], re.I | re.S)
      if existing != None:
        for i in self.shapes:
          if i.name == existing.group(1):
            newEntity.shapeReference = i
            newEntity.shapeLinked = True
            print "  Temp shape: %s" % i.name
            break
      else:
        vert = vertexPattern.search(wrlContent, startPos, endPos)
        ind = indexPattern.search(wrlContent, startPos, endPos)
        if vert != None and ind != None:
          newShape = shape()
          newShape.loadVertices(vert.group(1))
          newShape.loadPolygons(ind.group(1))
          tmp = re.search("DEF\s+(\w+)\s+Group", wrlContent[startPos:endPos], re.I | re.S)
          if tmp != None:
            newShape.name = tmp.group(1)
            print "  Shape: %s" % newShape.name
          mat = re.search("material\s+DEF\s+(\w+)\s+Material\s+{(.*?)}", wrlContent[startPos:endPos], re.I | re.S)
          if mat != None:
            newMaterial = material(mat.group(2))
            newMaterial.name = mat.group(1)
            self.materials.append(newMaterial)
            print "  Material: %s" % newMaterial.name
            newShape.materialReference = newMaterial
          mat = re.search("material\s+USE\s+(\w+)", wrlContent[startPos:endPos], re.I | re.S)
          if mat != None:
            for i in self.materials:
              if i.name == mat.group(1):
                newShape.materialReference = i
                newShape.materialLinked = True
                print "  Temp material: %s" % i.name
                break
          self.shapes.append(newShape)
          newEntity.shapeReference = newShape
      self.entities.append(newEntity)
      contentPos = endPos
  def saveFile(self, fileName):
    wrlFile = open(fileName, "w")
    wrlFile.write("#VRML V2.0 utf8\n#Exported from Blender by wrlconv.py\n")
    for ent in self.entities:
      ent.write(wrlFile)
    wrlFile.close()
  def render(self):
    for ent in self.entities:
      ent.render()

class entity:
  def __init__(self):
    self.name = ""
    self.shapeReference = None
    self.shapeLinked = False
    self.transform   = numpy.matrix([[1., 0., 0., 0.],
                                     [0., 1., 0., 0.],
                                     [0., 0., 1., 0.],
                                     [0., 0., 0., 1.]])
  def findTransform(self, string):
    translation = numpy.matrix([[1., 0., 0., 0.],
                                [0., 1., 0., 0.],
                                [0., 0., 1., 0.],
                                [0., 0., 0., 1.]])
    rotation    = numpy.matrix([[1., 0., 0., 0.],
                                [0., 1., 0., 0.],
                                [0., 0., 1., 0.],
                                [0., 0., 0., 1.]])
    scale       = numpy.matrix([[1., 0., 0., 0.],
                                [0., 1., 0., 0.],
                                [0., 0., 1., 0.],
                                [0., 0., 0., 1.]])
    tmp = re.search("translation\s+([+e\d\.\-]+)\s+([+e\d\.\-]+)\s+([+e\d\.\-]+)", string, re.I | re.S)
    if tmp != None:
      translation = numpy.matrix([[1., 0., 0., float(tmp.group(1))],
                                  [0., 1., 0., float(tmp.group(2))],
                                  [0., 0., 1., float(tmp.group(3))],
                                  [0., 0., 0., 1.]])
    tmp = re.search("rotation\s+([+e\d\.\-]+)\s+([+e\d\.\-]+)\s+([+e\d\.\-]+)\s+([+e\d\.\-]+)", string, re.I | re.S)
    if tmp != None:
      rotation = fillRotateMatrix([float(tmp.group(1)), float(tmp.group(2)), float(tmp.group(3))], float(tmp.group(4)));
    tmp = re.search("scale\s+([+e\d\.\-]+)\s+([+e\d\.\-]+)\s+([+e\d\.\-]+)", string, re.I | re.S)
    if tmp != None:
      scale = numpy.matrix([[float(tmp.group(1)), 0., 0., 0.],
                            [0., float(tmp.group(2)), 0., 0.],
                            [0., 0., float(tmp.group(3)), 0.],
                            [0., 0., 0., 1.]])
    self.transform = translation * rotation * scale
  def write(self, handler):
    print "Writing item %s" % self.name
    handler.write("DEF %s Transform {\n  children [\n" % self.name)
    if self.shapeReference != None:
      self.shapeReference.write(self.shapeLinked, self.transform, handler)
    handler.write("  ]\n}\n\n")
  def render(self):
    print "Drawing item %s" % self.name
    if self.shapeReference != None:
      self.shapeReference.render(self.transform)

class shape:
  def __init__(self):
    self.name = ""
    self.materialReference = None
    self.materialLinked = False
    self.vertices = []
    self.polygons = []
    self.polygonVertices = 0
    self.quadVertices = 0
    self.triangleVertices = 0
  def loadVertices(self, string):
    vectorPattern = re.compile("([+e\d\-\.]+)[ ,\t]+([+e\d\-\.]+)[ ,\t]+([+e\d\-\.]+).*?\n?", re.I | re.S)
    pos = 0
    while (1):
      tmp = vectorPattern.search(string, pos)
      if tmp == None:
        break
      self.vertices.append([float(tmp.group(1)), float(tmp.group(2)), float(tmp.group(3))])
      pos = tmp.end()
  def loadPolygons(self, string):
    polyPattern = re.compile("([ ,\t\d]+)-1.*?\n?", re.I | re.S)
    indPattern = re.compile("[ ,\t]*(\d+)[ ,\t]*", re.I | re.S)
    polyPos = 0
    while (1):
      poly = polyPattern.search(string, polyPos)
      if poly == None:
        break
      polyData = []
      indPos = 0
      while (1):
        ind = indPattern.search(poly.group(1), indPos)
        if ind == None:
          break
        polyData.append(int(ind.group(1)))
        indPos = ind.end()
      if len(polyData) == 3:
        self.triangleVertices += 3
      elif len(polyData) == 4:
        self.quadVertices += 4
      else:
        self.polygonVertices += len(polyData)
      self.polygons.append(polyData)
      polyPos = poly.end()
    print "Loaded tri: %d, quad: %d, poly: %d, total vert: %d" % (self.triangleVertices / 3, self.quadVertices / 4, len(self.polygons) - self.triangleVertices / 3 - self.quadVertices / 4, self.polygonVertices + self.triangleVertices + self.quadVertices)
  def write(self, link, transform, handler):
    handler.write("    Shape {\n")
    if self.materialReference != None:
      matLink = False
      if link == True or self.materialLinked == True:
        matLink = True
      self.materialReference.write(matLink, handler)
    handler.write("      geometry IndexedFaceSet {\n        coord Coordinate { point [\n")
    for i in range(0, len(self.vertices)):
      vert = numpy.matrix([[self.vertices[i][0]], [self.vertices[i][1]], [self.vertices[i][2]], [1.]])
      vert = transform * vert
      handler.write("          %f %f %f" % (float(vert[0]), float(vert[1]), float(vert[2])))
      if i != len(self.vertices) - 1:
        handler.write(",\n")
    handler.write(" ] }\n")
    handler.write("        coordIndex [\n")
    for i in range(0, len(self.polygons)):
      handler.write("          ")
      for j in range(0, len(self.polygons[i])):
        handler.write("%d, " % self.polygons[i][j])
      handler.write("-1")
      if i != len(self.polygons) - 1:
        handler.write(",\n")
    handler.write(" ]\n")
    handler.write("      }\n")
    handler.write("    }\n")
  def render(self, transform):
    if self.materialReference != None:
      self.materialReference.setMaterial()
    translatedVertices = []
    for i in range(0, len(self.vertices)):
      vert = numpy.matrix([[self.vertices[i][0]], [self.vertices[i][1]], [self.vertices[i][2]], [1.]])
      vert = transform * vert
      translatedVertices.append(numpy.matrix([[float(vert[0])], [float(vert[1])], [float(vert[2])]]))
    for i in range(0, len(self.polygons)):
      if len(self.polygons) == 3:
        glBegin(GL_TRIANGLES)
      elif len(self.polygons) == 4:
        glBegin(GL_QUADS)
      else:
        glBegin(GL_POLYGON)
      if len(self.polygons[i]) >= 3:
        normal = getNormal(translatedVertices[self.polygons[i][1]] - translatedVertices[self.polygons[i][0]], 
                           translatedVertices[self.polygons[i][0]] - translatedVertices[self.polygons[i][2]])
        normal /= -numpy.linalg.norm(normal)
        glNormal3f(normal[0], normal[1], normal[2])
      for j in range(0, len(self.polygons[i])):
        glVertex3f(translatedVertices[self.polygons[i][j]][0], translatedVertices[self.polygons[i][j]][1], translatedVertices[self.polygons[i][j]][2])
      glEnd()

class material:
  def __init__(self, string):
    self.name = ""
    self.diffuseColor = [0., 0., 0.]
    self.ambientIntensity = 0.
    self.specularColor = [0., 0., 0.]
    self.emissiveColor = [0., 0., 0.]
    self.shininess = 0.
    self.transparency = 0.
    tmp = re.search("diffuseColor\s+([+e\d\.]+)\s+([+e\d\.]+)\s+([+e\d\.]+)", string, re.I | re.S)
    if tmp != None:
      self.diffuseColor = [float(tmp.group(1)), float(tmp.group(2)), float(tmp.group(3))]
    tmp = re.search("ambientIntensity\s+([+e\d\.]+)", string, re.I | re.S)
    if tmp != None:
      self.ambientIntensity = float(tmp.group(1))
    self.ambientIntensity *= 3.
    if self.ambientIntensity > 1.:
      self.ambientIntensity = 1.
    tmp = re.search("specularColor\s+([+e\d\.]+)\s+([+e\d\.]+)\s+([+e\d\.]+)", string, re.I | re.S)
    if tmp != None:
      self.specularColor = [float(tmp.group(1)), float(tmp.group(2)), float(tmp.group(3))]
    tmp = re.search("emissiveColor\s+([+e\d\.]+)\s+([+e\d\.]+)\s+([+e\d\.]+)", string, re.I | re.S)
    if tmp != None:
      self.emissiveColor = [float(tmp.group(1)), float(tmp.group(2)), float(tmp.group(3))]
    tmp = re.search("shininess\s+([+e\d\.]+)", string, re.I | re.S)
    if tmp != None:
      self.shininess = float(tmp.group(1))
    tmp = re.search("transparency\s+([+e\d\.]+)", string, re.I | re.S)
    if tmp != None:
      self.transparency = float(tmp.group(1))
  def write(self, link, handler):
    if link == False:
      handler.write("      appearance Appearance {\n        material DEF %s Material {\n" % self.name)
      handler.write("          diffuseColor %f %f %f\n" % (self.diffuseColor[0], self.diffuseColor[1], self.diffuseColor[2]))
      handler.write("          emissiveColor %f %f %f\n" % (self.emissiveColor[0], self.emissiveColor[1], self.emissiveColor[2]))
      handler.write("          specularColor %f %f %f\n" % (self.specularColor[0], self.specularColor[1], self.specularColor[2]))
      handler.write("          ambientIntensity %f\n" % self.ambientIntensity)
      handler.write("          transparency %f\n" % self.transparency)
      handler.write("          shininess %f\n" % self.shininess)
      handler.write("        }\n      }\n")
    else:
      handler.write("      appearance Appearance {\n        material USE %s\n      }\n" % self.name)
  def setMaterial(self):
    glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
    glColor4f(self.diffuseColor[0] * self.ambientIntensity,
              self.diffuseColor[1] * self.ambientIntensity,
              self.diffuseColor[2] * self.ambientIntensity,
              1.0 - self.transparency)
#    glColorMaterial(GL_FRONT_AND_BACK, GL_SPECULAR)
#    glColor3f(self.specularColor[0], self.specularColor[1], self.specularColor[2])
#    glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)

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
    self.vertexList  = []
    self.normalList  = []
    self.arrayOffset = []
    self.arraySize   = []
    self.vertexVBO   = []
    self.normalVBO   = []
    self.cntr = time.time()
    self.fps = 0
    self.mts = []
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
  def drawAxis(self):
    glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
    glColor3f(1., 0., 0.)
    glBegin(GL_LINES)
    glNormal3f(0., 0., 1.)
    glVertex3f(0., 0., 0.)
    glVertex3f(4., 0., 0.)
    glEnd()
    glColor3f(0., 1., 0.)
    glBegin(GL_LINES)
    glNormal3f(0., 0., 1.)
    glVertex3f(0., 0., 0.)
    glVertex3f(0., 4., 0.)
    glEnd()
    glColor3f(0., 0., 1.)
    glBegin(GL_LINES)
    glNormal3f(1., 0., 0.)
    glVertex3f(0., 0., 0.)
    glVertex3f(0., 0., 2.)
    glEnd()
  def drawScene(self):
    self.updated = True
    if self.updated == True:
      self.updated = False
      glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
      glLoadIdentity()
      gluLookAt(float(self.camera[0]), float(self.camera[1]), float(self.camera[2]), 
                float(self.pov[0]), float(self.pov[1]), float(self.pov[2]), 0., 0., 1.)
      glLightfv(GL_LIGHT0, GL_POSITION, self.light)
      glEnableClientState(GL_NORMAL_ARRAY)
      glEnableClientState(GL_VERTEX_ARRAY)
      for i in range(0, len(self.vertexList)):
        if self.mts[i] != None:
          self.mts[i].setMaterial()
        glBindBuffer(GL_ARRAY_BUFFER, self.vertexVBO[i])
        glVertexPointer(3, GL_FLOAT, 0, None)
        glBindBuffer( GL_ARRAY_BUFFER, self.normalVBO[i])
        glNormalPointer(GL_FLOAT, 0, None)
        #glVertexPointer(3, GL_FLOAT, 0, self.vertexList)
        #glNormalPointer(GL_FLOAT, 0, self.normalList)
        #glDrawArrays(GL_QUADS, 0, len(self.vertexList[i]) / 4)
        glDrawArrays(GL_TRIANGLES, self.arrayOffset[i][0], self.arraySize[i][0])
        glDrawArrays(GL_QUADS, self.arrayOffset[i][1], self.arraySize[i][1])
        glDrawArrays(GL_POLYGON, self.arrayOffset[i][2], self.arraySize[i][2])
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
    gluPerspective(45.0, float(self.width)/float(self.height), 0.1, 100.0)
    glMatrixMode(GL_MODELVIEW)
    self.updated = True
  def initScene(self, aScene):
#    sh1 = aScene.shapes[0]
    for sh in aScene.shapes:
      self.mts.append(sh.materialReference)
      tsa = time.time()
      #length = len(sh.triangles) * 9 + len(sh.quads) * 12 + sh.polygonVertices * 3
      length = (sh.triangleVertices + sh.quadVertices + sh.polygonVertices) * 3
      self.vertexList.append(numpy.zeros(length, dtype = numpy.float32))
      self.normalList.append(numpy.zeros(length, dtype = numpy.float32))
      self.arrayOffset.append([])
      self.arraySize.append([])
      listID = len(self.vertexList) - 1
      self.arrayOffset[listID].append(0)
      self.arraySize[listID].append(sh.triangleVertices)
      self.arrayOffset[listID].append(sh.triangleVertices)
      self.arraySize[listID].append(sh.quadVertices)
      self.arrayOffset[listID].append(sh.triangleVertices + sh.quadVertices)
      self.arraySize[listID].append(sh.polygonVertices)
      tPos = 0
      qPos = sh.triangleVertices * 3
      pPos = sh.triangleVertices * 3 + sh.quadVertices * 4
      for poly in sh.polygons:
        v0 = numpy.array(sh.vertices[poly[0]])
        v1 = numpy.array(sh.vertices[poly[1]])
        v2 = numpy.array(sh.vertices[poly[2]])
        normal = getNormal(v1 - v0, v0 - v2)
        det = numpy.linalg.norm(normal)
        if det != 0:
          normal /= -det
        pos = 0
        if len(poly) == 3:
          pos = tPos
          tPos += 9
        elif len(poly) == 4:
          pos = qPos
          qPos += 12
        else:
          pos = pPos
          pPos += len(poly) * 3
        for vert in poly:
          self.vertexList[listID][pos]     = sh.vertices[vert][0]
          self.vertexList[listID][pos + 1] = sh.vertices[vert][1]
          self.vertexList[listID][pos + 2] = sh.vertices[vert][2]
          self.normalList[listID][pos]     = normal[0]
          self.normalList[listID][pos + 1] = normal[1]
          self.normalList[listID][pos + 2] = normal[2]
          pos += 3
      print "Count tri: %d, quad: %d, poly: %d, total vert: %d" % (self.arraySize[listID][0] / 3, self.arraySize[listID][1] / 4, len(sh.polygons) - self.arraySize[listID][0] / 3 - self.arraySize[listID][1] / 4, length / 3)
      #print "Size tri: %d, quad: %d, poly: %d" % (self.arraySize[listID][0], self.arraySize[listID][1], self.arraySize[listID][2])
      #print "Offset tri: %d, quad: %d, poly: %d" % (self.arrayOffset[listID][0], self.arrayOffset[listID][1], self.arrayOffset[listID][2])
      tsb = time.time()
      print "Delta: %f" % (tsb - tsa)
      self.vertexVBO.append(glGenBuffers(1))
      #self.vertexVBO[listID] = glGenBuffers(1)
      glBindBuffer(GL_ARRAY_BUFFER, self.vertexVBO[listID])
      glBufferData(GL_ARRAY_BUFFER, self.vertexList[listID], GL_STATIC_DRAW)
      self.normalVBO.append(glGenBuffers(1))
      #self.normalVBO[listID] = glGenBuffers(1)
      glBindBuffer(GL_ARRAY_BUFFER, self.normalVBO[listID])
      glBufferData(GL_ARRAY_BUFFER, self.normalList[listID], GL_STATIC_DRAW)
    #glEnableClientState(GL_COLOR_ARRAY)

    #self.drawList = glGenLists(1)
    ##Draw scene to the list 1
    #glNewList(self.drawList, GL_COMPILE)
    ##Draw axis
    #self.drawAxis()
    ##Draw objects
    #aScene.render()
    #glEndList()
  def initGraphics(self):
    glClearColor(0.0, 0.0, 0.0, 0.0)
    glClearDepth(1.0)
    glDepthFunc(GL_LESS)
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_LIGHTING)
    glEnable(GL_LIGHT0)
    glLightfv(GL_LIGHT0, GL_POSITION, self.light)
    glEnable(GL_COLOR_MATERIAL)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45.0, float(self.width)/float(self.height), 0.1, 1000.0)
    glMatrixMode(GL_MODELVIEW)
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
      rotMatrixA = numpy.matrix([[math.cos(zrot), -math.sin(zrot), 0., 0.],
                                 [math.sin(zrot),  math.cos(zrot), 0., 0.],
                                 [            0.,              0., 1., 0.],
                                 [            0.,              0., 0., 1.]])
      rotMatrixB = fillRotateMatrix(normal, (yPos - self.mousePos[1]) / 100.)
      self.camera = rotMatrixA * rotMatrixB * self.camera
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

parser = OptionParser()
parser.add_option("-v", "--view", dest="view", help="Render and show model.", default=False, action="store_true")
parser.add_option("-t", "--translate", dest="translate", help="Move shapes to new coordinates [x,y,z], default value \"0.,0.,0.\".", default='0.,0.,0.')
parser.add_option("-r", "--rotate", dest="rotate", help="Rotate shapes around vector [x,y,z] by angle, default value \"0.,0.,1.,0.\".", default='0.,0.,1.,0.')
parser.add_option("-s", "--scale", dest="scale", help="Scale shapes by [x,y,z], default value \"1.,1.,1.\".", default='1.,1.,1.')
(options, args) = parser.parse_args()

if len(args) > 0:
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

  sc = scene()
  sc.setTransform(gTranslate, gRotate, gScale)
  for fname in args:
    sc.loadFile(fname)
    if options.view == False:
      sc.saveFile(fname + "~")
      sc.clear()
  if options.view == True:
    rend = render(sc)
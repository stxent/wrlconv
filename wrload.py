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

def fillRotateMatrix(ax, ay, az, angle):
  a0 = math.cos(angle) + ax*ax*(1 - math.cos(angle))
  a1 = ax*ay*(1 - math.cos(angle)) - az*math.sin(angle)
  a2 = ax*az*(1 - math.cos(angle)) + ay*math.sin(angle)
  b0 = ay*ax*(1 - math.cos(angle)) + az*math.sin(angle)
  b1 = math.cos(angle) + ay*ay*(1 - math.cos(angle))
  b2 = ay*az*(1 - math.cos(angle)) - ax*math.sin(angle)
  c0 = az*ax*(1 - math.cos(angle)) - ay*math.sin(angle)
  c1 = az*ay*(1 - math.cos(angle)) + ax*math.sin(angle)
  c2 = math.cos(angle) + az*az*(1 - math.cos(angle))
  return numpy.matrix([[a0, a1, a2, 0.],
                       [b0, b1, b2, 0.],
                       [c0, c1, c2, 0.],
                       [0., 0., 0., 1.]])

def getVector(p1, p2):
  res = [0., 0., 0.]
  res[0] = p1[0] - p2[0]
  res[1] = p1[1] - p2[1]
  res[2] = p1[2] - p2[2]
  return res

def getNormal(v1, v2):
  res = [0., 0., 0.]
  res[0] = v1[1] * v2[2] - v1[2] * v2[1]
  res[1] = v1[2] * v2[0] - v1[0] * v2[2]
  res[2] = v1[0] * v2[1] - v1[1] * v2[0]
  return res

def normalize(vect):
  mag = math.sqrt(vect[0]*vect[0] + vect[1]*vect[1] + vect[2]*vect[2])
  if mag == 0:
    return [0., 0., 0.]
  else:
    return [vect[0] / mag, vect[1] / mag, vect[2] / mag]

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

totalPolygons = 0

class scene:
  def __init__(self):
    self.materials = []
    self.entities = []
    self.shapes = []
  def loadFile(self, fileName):
    wrlFile = open(fileName, "r")
    wrlContent = wrlFile.read()
    wrlFile.close()
    contentPos = 0
    transformPattern = re.compile("DEF\s+(\w+)\s+Transform\s+{", re.I | re.S)
    #vertexPattern = re.compile("coord[\s\w]+{\s+point\s+\[", re.I | re.S)
    #indexPattern = re.compile("coordIndex[\s]+\[", re.I | re.S)
    vertexPattern = re.compile("coord[\s\w]+{\s+point\s+\[([e,\s\d\.\-]*)\]\s+}", re.I | re.S)
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
      print 'Object %s at %d' % (newEntity.name, contentPos)
      newEntity.findTransform(wrlContent[startPos:endPos])
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
          #vertEnd = getChunkEnd(wrlContent, vert.end())
          #indEnd = getChunkEnd(wrlContent, ind.end())
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
    wrlFile.write("#VRML V2.0 utf8\n#Exported from Blender with wrlconv.py\n")
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
    self.translation = numpy.matrix([[1., 0., 0., 0.],
                                     [0., 1., 0., 0.],
                                     [0., 0., 1., 0.],
                                     [0., 0., 0., 1.]])
    self.rotation    = numpy.matrix([[1., 0., 0., 0.],
                                     [0., 1., 0., 0.],
                                     [0., 0., 1., 0.],
                                     [0., 0., 0., 1.]])
    self.scale       = numpy.matrix([[1., 0., 0., 0.],
                                     [0., 1., 0., 0.],
                                     [0., 0., 1., 0.],
                                     [0., 0., 0., 1.]])
  def findTransform(self, string):
    tmp = re.search("translation\s+([e\d\.\-]+)\s+([e\d\.\-]+)\s+([e\d\.\-]+)", string, re.I | re.S)
    if tmp != None:
      self.translation = numpy.matrix([[1., 0., 0., float(tmp.group(1))],
                                       [0., 1., 0., float(tmp.group(2))],
                                       [0., 0., 1., float(tmp.group(3))],
                                       [0., 0., 0., 1.]])
    tmp = re.search("rotation\s+([e\d\.\-]+)\s+([e\d\.\-]+)\s+([e\d\.\-]+)\s+([e\d\.\-]+)", string, re.I | re.S)
    if tmp != None:
      self.rotation = fillRotateMatrix(float(tmp.group(1)), float(tmp.group(2)), float(tmp.group(3)), float(tmp.group(4)));
    tmp = re.search("scale\s+([e\d\.\-]+)\s+([e\d\.\-]+)\s+([e\d\.\-]+)", string, re.I | re.S)
    if tmp != None:
      self.scale = numpy.matrix([[float(tmp.group(1)), 0., 0., 0.],
                                 [0., float(tmp.group(2)), 0., 0.],
                                 [0., 0., float(tmp.group(3)), 0.],
                                 [0., 0., 0., 1.]])
  def write(self, handler):
    print "Writing entry %s" % self.name
    handler.write("DEF %s Transform {\n  children [\n" % self.name)
    matr = self.translation * self.rotation * self.scale
    if self.shapeReference != None:
      self.shapeReference.write(self.shapeLinked, matr, handler)
    handler.write("  ]\n}\n\n")
  def render(self):
    print "Drawing entry %s" % self.name
    matr = self.translation * self.rotation * self.scale
    if self.shapeReference != None:
      self.shapeReference.render(matr)

class shape:
  def __init__(self):
    self.name = ""
    self.materialReference = None
    self.materialLinked = False
    self.vertices = []
    self.polygons = []
  def loadVertices(self, string):
    vectorPattern = re.compile("([e\d\-\.]+)[ ,\t]+([e\d\-\.]+)[ ,\t]+([e\d\-\.]+)[ ,\t]*\n", re.I | re.S)
    pos = 0
    while (1):
      tmp = vectorPattern.search(string, pos)
      if tmp == None:
        break
      self.vertices.append([float(tmp.group(1)), float(tmp.group(2)), float(tmp.group(3))])
#      print self.vertices[len(self.vertices) - 1]
      pos = tmp.end()
  def loadPolygons(self, string):
    polyPattern = re.compile("([ ,\t\d]+)-1[ ,\t]*\n", re.I | re.S)
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
      self.polygons.append(polyData)
#      print polyData
      polyPos = poly.end()
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
    global totalPolygons
    if self.materialReference != None:
      self.materialReference.setMaterial()
    translatedVertices = []
    for i in range(0, len(self.vertices)):
      vert = numpy.matrix([[self.vertices[i][0]], [self.vertices[i][1]], [self.vertices[i][2]], [1.]])
      vert = transform * vert
      translatedVertices.append([float(vert[0]), float(vert[1]), float(vert[2])])
    for i in range(0, len(self.polygons)):
      if len(self.polygons) == 3:
        glBegin(GL_TRIANGLES)
      elif len(self.polygons) == 4:
        glBegin(GL_QUADS)
      else:
        glBegin(GL_POLYGON)
      if len(self.polygons[i]) >= 3:
        normal = normalize(getNormal(getVector(translatedVertices[self.polygons[i][1]], translatedVertices[self.polygons[i][0]]), 
                                     getVector(translatedVertices[self.polygons[i][0]], translatedVertices[self.polygons[i][2]])))
        glNormal3f(-normal[0], -normal[1], -normal[2])
      for j in range(0, len(self.polygons[i])):
        glVertex3f(translatedVertices[self.polygons[i][j]][0], translatedVertices[self.polygons[i][j]][1], translatedVertices[self.polygons[i][j]][2])
      glEnd()
      totalPolygons += 1

class material:
  def __init__(self, string):
    self.name = ""
    self.diffuseColor = [0., 0., 0.]
    self.ambientIntensity = 0.
    self.specularColor = [0., 0., 0.]
    self.emissiveColor = [0., 0., 0.]
    self.shininess = 0.
    self.transparency = 0.
    tmp = re.search("diffuseColor\s+([e\d\.]+)\s+([e\d\.]+)\s+([e\d\.]+)", string, re.I | re.S)
    if tmp != None:
      self.diffuseColor = [float(tmp.group(1)), float(tmp.group(2)), float(tmp.group(3))]
    tmp = re.search("ambientIntensity\s+([e\d\.]+)", string, re.I | re.S)
    if tmp != None:
      self.ambientIntensity = float(tmp.group(1))
    self.ambientIntensity *= 3.
    tmp = re.search("specularColor\s+([e\d\.]+)\s+([e\d\.]+)\s+([e\d\.]+)", string, re.I | re.S)
    if tmp != None:
      self.specularColor = [float(tmp.group(1)), float(tmp.group(2)), float(tmp.group(3))]
    tmp = re.search("emissiveColor\s+([e\d\.]+)\s+([e\d\.]+)\s+([e\d\.]+)", string, re.I | re.S)
    if tmp != None:
      self.emissiveColor = [float(tmp.group(1)), float(tmp.group(2)), float(tmp.group(3))]
    tmp = re.search("shininess\s+([e\d\.]+)", string, re.I | re.S)
    if tmp != None:
      self.shininess = float(tmp.group(1))
    tmp = re.search("transparency\s+([e\d\.]+)", string, re.I | re.S)
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

sc = scene()
updated = True
camPos = numpy.matrix([[0.], [20.], [20.], [1.]])
lightPos = [20., 20., 20., 1.]

def drawScene():
  global sc
  global camPos
  global updated
  global lightPos
  if updated == True:
    updated = False
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glLoadIdentity()
    glLightfv(GL_LIGHT0, GL_POSITION, lightPos)
    gluLookAt(float(camPos[0]), float(camPos[1]), float(camPos[2]), 0., 0., 0., 0., 0., 1.)
    glCallList(1)
    glutSwapBuffers()
  else:
    time.sleep(.01)

def resizeGL(width, height):
  global updated
  if height == 0:
    height = 1
  glViewport(0, 0, width, height)
  glMatrixMode(GL_PROJECTION)
  glLoadIdentity()
  gluPerspective(45.0, float(width)/float(height), 0.1, 100.0)
  glMatrixMode(GL_MODELVIEW)
  updated = True

def initGL(width, height):
  global lightPos
  global totalPolygons
  glClearColor(0.0, 0.0, 0.0, 0.0)
  glClearDepth(1.0)
  glDepthFunc(GL_LESS)
  glEnable(GL_DEPTH_TEST)
  #glLightfv(GL_LIGHT0, GL_POSITION, [20., 20., 20., 1.])
  #glEnable(GL_LIGHT0)
  glLightfv(GL_LIGHT0, GL_POSITION, lightPos)
  glEnable(GL_LIGHT0)
  glEnable(GL_LIGHTING)
  glEnable(GL_COLOR_MATERIAL)
  glMatrixMode(GL_PROJECTION)
  glLoadIdentity()
  gluPerspective(45.0, float(width)/float(height), 0.1, 1000.0)
  glMatrixMode(GL_MODELVIEW)
#Draw scene in the list 1
  glNewList(1, GL_COMPILE)
#Draw axis
  glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
  glColor3f(1., 1., 1.)
  glBegin(GL_LINES)
  glNormal3f(0., 0., 1.)
  glVertex3f(0., 0., 0.)
  glVertex3f(4., 0., 0.)
  glVertex3f(0., 0., 0.)
  glVertex3f(0., 4., 0.)
  glNormal3f(1., 0., 0.)
  glVertex3f(0., 0., 0.)
  glVertex3f(0., 0., 2.)
  glEnd()
#Draw objects
  sc.render()
  glEndList()
  print "Total polygons: %d" % totalPolygons

def keyPressed(*args):
  global lightPos
  global updated
  if args[0] == "w":
    lightPos[1] -= 1.
  if args[0] == "s":
    lightPos[1] += 1.
  if args[0] == "d":
    lightPos[0] -= 1.
  if args[0] == "a":
    lightPos[0] += 1.
  if args[0] == "[":
    lightPos[2] -= 1.
  if args[0] == "]":
    lightPos[2] += 1.
  print lightPos
  updated = True

mouseDrag = False
mousePos = [0., 0.]

def mouseButton(*args):
  global mouseDrag, mousePos, camPos
  global updated
  if args[0] == GLUT_LEFT_BUTTON:
    if args[1] == GLUT_DOWN:
      mouseDrag = True
      mousePos = [args[2], args[3]]
    else:
      mouseDrag = False
  if args[0] == 3 and args[1] == GLUT_DOWN:
    zm = 0.9
    scaleMatrix = numpy.matrix([[zm, 0., 0., 0.],
                                [0., zm, 0., 0.],
                                [0., 0., zm, 0.],
                                [0., 0., 0., 1.]])
    camPos = scaleMatrix * camPos
  if args[0] == 4 and args[1] == GLUT_DOWN:
    zm = 1.1
    scaleMatrix = numpy.matrix([[zm, 0., 0., 0.],
                                [0., zm, 0., 0.],
                                [0., 0., zm, 0.],
                                [0., 0., 0., 1.]])
    camPos = scaleMatrix * camPos
  updated = True

def mouseMove(*args):
  global mouseDrag, mousePos, camPos
  global updated
  if mouseDrag == True:
    normal = normalize(getNormal([float(camPos[0]), float(camPos[1]), float(camPos[2])], [0., 0., 1.]))
    zrot = (mousePos[0] - args[0]) / 100.
    rotMatrixA = numpy.matrix([[math.cos(zrot), -math.sin(zrot), 0., 0.],
                              [math.sin(zrot),  math.cos(zrot), 0., 0.],
                              [            0.,              0., 1., 0.],
                              [            0.,              0., 0., 1.]])
    rotMatrixB = fillRotateMatrix(normal[0], normal[1], normal[2], (args[1] - mousePos[1]) / 100.)
    camPos = rotMatrixA * rotMatrixB * camPos
    mousePos = [args[0], args[1]]
  updated = True

parser = OptionParser()
parser.add_option("-r", "--render", dest="render", help="Render model.", default=False, action="store_true")
#parser.add_option("-o", "--output", dest="outpath", help="Output folder.", default='.')
(options, args) = parser.parse_args()

for fname in args:
  sc.loadFile(fname)
#  sc.saveFile(fname + "~")

if options.render == True:
  glutInit(sys.argv)
  glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_ALPHA | GLUT_DEPTH)
  glutInitWindowSize(800, 800)
  glutInitWindowPosition(0, 0)
  glutCreateWindow("VRML view")
  glutDisplayFunc(drawScene)
  glutIdleFunc(drawScene)
  glutReshapeFunc(resizeGL)
  glutKeyboardFunc(keyPressed)
  glutMotionFunc(mouseMove)
  glutMouseFunc(mouseButton)
  initGL(800, 800)
  glutMainLoop()
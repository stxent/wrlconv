#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# wrload.py
# Copyright (C) 2011 xent
# Project is distributed under the terms of the GNU General Public License v3.0

import argparse
import math
import os
import re
import sys
import numpy as np

import helpers
import render_ogl41
import vrml_export
import vrml_export_kicad
import vrml_import
import x3d_export
import x3d_import

def export_mesh(meshes, output_path, kicad_compat):
    extension = os.path.splitext(output_path)[1][1:].lower()
    if extension == 'wrl':
        if kicad_compat:
            vrml_export_kicad.store(meshes, output_path)
        else:
            vrml_export.store(meshes, output_path)
    elif extension == 'x3d':
        x3d_export.store(meshes, output_path)

def import_meshes(filenames, filter_string=''):
    export_list = []

    for name in filenames:
        extension = os.path.splitext(name)[1][1:].lower()
        if extension == 'wrl':
            export_list.extend(vrml_import.load(name))
        elif extension == 'x3d':
            export_list.extend(x3d_import.load(name))

    if filter_string != '':
        export_list = filter(lambda mesh: re.search(filter_string, mesh.ident, re.S) is not None,
                             export_list)
    return export_list

def transform_meshes(meshes, translation, rotation, scaling, options):
    for mesh in meshes:
        mesh.transform.rotate(rotation[0:3], rotation[3])
        mesh.transform.scale(scaling)
        mesh.transform.translate(translation)

        if options.normals:
            mesh.appearance().normals = True
        if options.smooth:
            mesh.appearance().smooth = True

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-q', dest='quiet', help='disable model rendering',
                        default=False, action='store_true')
    parser.add_argument('-o', dest='output', help='write transformed models to a specified file',
                        default='')
    parser.add_argument('-t', dest='translation', help='move the mesh to new coordinates',
                        default='0,0,0')
    parser.add_argument('-r', dest='rotation', help='rotate the mesh around a vector',
                        default='0,0,1,0')
    parser.add_argument('-s', dest='scale', help='scale the shapes by the specified factors',
                        default='1,1,1')
    parser.add_argument('-f', dest='filter', help='filter objects by name using regular expression',
                        default='')
    parser.add_argument('-d', dest='debug', help='show debug information',
                        default=False, action='store_true')
    parser.add_argument('--kicad', dest='kicad', help='use VRML format with simplified syntax',
                        default=False, action='store_true')
    parser.add_argument('--axes', dest='axes', help='display coordinate axes in the 3D viewer',
                        default=False, action='store_true')
    parser.add_argument('--grid', dest='grid', help='display a grid in the 3D viewer',
                        default=False, action='store_true')
    parser.add_argument('--fast', dest='fast', help='disable visual effects',
                        default=False, action='store_true')
    parser.add_argument('--normals', dest='normals', help='Visualize surface normals of the mesh',
                        default=False, action='store_true')
    parser.add_argument('--overlay', dest='overlay', help='enable a text overlay',
                        default=False, action='store_true')
    parser.add_argument('--smooth', dest='smooth', help='enable smooth shading',
                        default=False, action='store_true')
    parser.add_argument(dest='files', nargs='*')
    options = parser.parse_args()

    translation = np.zeros(3)
    rotation = np.array([0.0, 0.0, 1.0, 0.0])
    scaling = np.ones(3)

    try:
        rotation = np.array([float(x) for x in options.rotation.split(',')[0:4]])
        rotation[3] = math.radians(rotation[3])
        scaling = np.array([float(x) for x in options.scale.split(',')[0:3]])
        translation = np.array([float(x) for x in options.translation.split(',')[0:3]])
    except ValueError:
        print('Wrong argument')
        sys.exit()

    if options.debug:
        render_ogl41.DEBUG_ENABLED = True
        vrml_export.DEBUG_ENABLED = True
        vrml_export_kicad.DEBUG_ENABLED = True
        vrml_import.DEBUG_ENABLED = True
        x3d_import.DEBUG_ENABLED = True
        x3d_export.DEBUG_ENABLED = True

    export_list = import_meshes(options.files, options.filter)
    transform_meshes(export_list, translation, rotation, scaling, options)

    if options.output != '':
        export_mesh(export_list, options.output, options.kicad)

    if not options.quiet:
        effects = {} if options.fast else {'overlay': options.overlay, 'antialiasing': 4}
        helper_objects = []
        if options.grid:
            helper_objects += helpers.make_grid()
        if options.axes:
            helper_objects += helpers.make_axes()
        render = render_ogl41.Render(helper_objects + export_list, effects)
        render.run()

if __name__ == '__main__':
    main()

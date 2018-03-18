Installation
------------

Scripts require Python 2.7 or Python 3 with NumPy, OpenGL, Pillow and lxml packages.
Model loader supports direct import from FreeCAD and import from Blender through X3D format. For Blender "Export X3D Hierarchy" option should be disabled.

Quickstart
----------

View model in internal viewer, show grid:
```sh
wrload.py --grid -v FILE
```

Load model from INPUT.x3d and save it to OUTPUT.wrl with format change from X3D to VRML, similar to Wings3D:
```sh
wrload.py INPUT.x3d -o OUTPUT.wrl
```

Show help message and exit:
```sh
wrload.py -h
```

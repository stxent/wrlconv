# 3D Model Loader and Viewer

![Python 3.6](https://img.shields.io/badge/Python-3.6-blue.svg)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

A script for loading, transforming, and viewing 3D models with support for
format conversion.

## System Requirements

* **Python 3** (with the following packages — see *Required Packages* below)
* **OpenGL 4.1** or higher (required for rendering features and visual effects)

## Required Packages

The script requires the following Python packages:

* `glfw`
* `lxml` (for parsing XML‑based formats like X3D)
* `numpy`
* `Pillow` (for texture and image handling)
* `PyOpenGL`

## Installation

Ensure you have Python 3 and the required packages installed.
Then, create a symbolic link to the script in your user‑specific `bin`
directory (typically `~/bin/`), or add the script’s directory to your `PATH`
environment variable:

Supported Formats and Features

* Import from Blender via X3D format (note: the Export X3D Hierarchy option
  must be disabled in Blender)
* Format conversion: X3D ↔ VRML (including simplified VRML for KiCad
  similar to Wings3D)
* Real‑time 3D rendering with OpenGL 4.1 features
* Mesh transformations: translation, rotation, scaling
* Visualization options: grid, axes, normals, smooth shading

## Quickstart

View a model in the internal viewer and show the grid:

```sh
wrload.py --grid FILE
```

Load a model from INPUT.x3d and save it to OUTPUT.wrl with format conversion
from X3D to VRML, disable real-time viewer:

```sh
wrload.py -q INPUT.x3d -o OUTPUT.wrl
```

## Script Options

The script supports the following command‑line options:

| Option | Description | Default value |
| ------ | ----------- | ------------- |
| -q, --quiet | Disable model rendering (run in silent mode) | False |
| -o, --output | Write transformed models to a specified file | Empty string |
| -t, --translation | Move the mesh to new coordinates (x,y,z). Specify three comma‑separated values for the translation vector | (0,0,0) |
| -r, --rotation | Rotate the mesh around a vector (x,y,z) by a given angle in degrees. Format: x,y,z,angle (three coordinates for the rotation axis and the rotation angle) | (0,0,1,0) |
| -s, --scale | Scale the shapes by the specified factors along the X, Y, and Z axes (x,y,z). Provide three comma‑separated scale factors | (1,1,1) |
| -f, --filter | Filter objects by name using a regular expression. Only objects whose names match the pattern will be processed | Empty string |
| -d, --debug | Enable debug mode: show detailed information about the processing steps and potential issues | False |
| --kicad | Use VRML format with simplified syntax, suitable for integration with KiCad | False |
| --axes | Display coordinate axes in the 3D viewer to help with orientation | False |
| --grid | Display a grid in the 3D viewer for better spatial reference | False |
| --fast | Disable visual effects (such as shadows or reflections) to achieve faster rendering. Useful for complex models | False |
| --normals | Visualize surface normals of the mesh. This helps in debugging geometry and lighting issues | False |
| --smooth | Enable smooth shading for a more realistic appearance of curved surfaces | False |

## Examples

Apply translation and scale:

```sh
wrload.py model.x3d -t 10.0,5.0,0.0 -s 2.0,2.0,1.0
```

Run quietly, rotate around Y‑axis by 45° and save as VRML:

```sh
wrload.py -q input.x3d --rotation 0.0,1.0,0.0,45.0 -o output.wrl
```

Run with debug info and show normals:

```sh
wrload.py -d --normals model.wrl
```

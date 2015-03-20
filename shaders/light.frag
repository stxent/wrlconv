#version 410 core
#extension GL_ARB_texture_rectangle : enable
#define LIGHT_COUNT 2

uniform float dataSize;

in float height;
in float srcOffset;

layout(location = 0) out vec4 color;

void main(void)
{
  color = vec4(height, height, height, 1.0);

  if (srcOffset >= dataSize)
    color.a = 0.0;
}

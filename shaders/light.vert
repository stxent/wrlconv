#version 410 core

layout(location = 0) in vec3 position;
layout(location = 2) in vec2 texel;

out vec2 rawTexel;

void main(void)
{
  rawTexel = texel;
  gl_Position = vec4(position, 1.0);
}

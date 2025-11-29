#version 300 es

uniform mat4 projectionMatrix;
uniform mat4 modelViewMatrix;
uniform mat4 normalMatrix;

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 color;

out vec3 calcPosition;
out vec3 calcColor;

void main(void)
{
  vec4 pos = modelViewMatrix * vec4(position, 1.0);

  calcPosition = pos.xyz;
  calcColor = color;

  gl_Position = projectionMatrix * pos;
}

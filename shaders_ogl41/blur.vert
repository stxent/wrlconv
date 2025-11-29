#version 300 es

uniform mat4 projectionMatrix;
uniform mat4 modelViewMatrix;

out vec2 texel;

layout(location = 0) in vec3 position;

void main(void)
{
  vec4 pos = modelViewMatrix * vec4(position, 1.0);

  texel = (pos.xy + 1.0) / 2.0;
  gl_Position = projectionMatrix * pos;
}

#version 330 core

uniform mat4 projectionMatrix;
uniform mat4 modelViewMatrix;

#ifndef AA_SAMPLES
out vec2 texel;
#endif

layout(location = 0) in vec3 position;

void main(void)
{
  vec4 pos = modelViewMatrix * vec4(position, 1.0);

#ifndef AA_SAMPLES
  texel = (pos.xy + 1.0) / 2.0;
#endif

  gl_Position = projectionMatrix * pos;
}

#version 300 es
precision mediump float;

in vec3 calcPosition;
in vec3 calcColor;

layout(location = 0) out vec4 color;

void main(void)
{
  vec3 view = normalize(-calcPosition.xyz);
  color = vec4(calcColor, 1.0);
  color = clamp(color, 0.0, 1.0);
}

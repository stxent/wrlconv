#version 410 core

in vec2 texel;

layout(location = 0) out vec4 color;

void main(void)
{
  color.rgb = vec3(texel.y / 2.0 + 0.25);
  color.b += 0.15;
  color.a = 1.0;

  color = clamp(color, 0.0, 1.0);
}

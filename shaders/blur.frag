#version 330 core

uniform sampler2D colorTexture;
uniform sampler2D maskTexture;

uniform vec2 direction;

in vec2 texel;

layout(location = 0) out vec4 color;

vec4 blur(sampler2D image, vec2 uv, vec2 dir)
{
  vec2 offset1 = vec2(1.4117647058823530) * dir;
  vec2 offset2 = vec2(3.2941176470588234) * dir;
  vec2 offset3 = vec2(5.1764705882352940) * dir;
  vec4 color = vec4(0.0);

  color += texture2D(image, uv) * 0.1964825501511404;
  color += texture2D(image, uv + offset1) * 0.2969069646728344;
  color += texture2D(image, uv - offset1) * 0.2969069646728344;
  color += texture2D(image, uv + offset2) * 0.09447039785044732;
  color += texture2D(image, uv - offset2) * 0.09447039785044732;
  color += texture2D(image, uv + offset3) * 0.010381362401148057;
  color += texture2D(image, uv - offset3) * 0.010381362401148057;

  return color;
}

void main(void)
{
  color = blur(colorTexture, texel, direction);
}

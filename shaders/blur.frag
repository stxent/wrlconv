#version 330 core

uniform sampler2D colorTexture;

#ifdef MASKED
uniform sampler2D maskTexture;
uniform sampler2D sourceTexture;
#endif

uniform vec2 direction;

in vec2 texel;

layout(location = 0) out vec4 color;

vec3 blur(sampler2D image, vec2 uv, vec2 dir)
{
  vec2 offset1 = vec2(1.4117647058823530) * dir;
  vec2 offset2 = vec2(3.2941176470588234) * dir;
  vec2 offset3 = vec2(5.1764705882352940) * dir;
  vec3 color = vec3(0.0);

  color += texture2D(image, uv).rgb * 0.1964825501511404;
  color += texture2D(image, uv + offset1).rgb * 0.2969069646728344;
  color += texture2D(image, uv - offset1).rgb * 0.2969069646728344;
  color += texture2D(image, uv + offset2).rgb * 0.09447039785044732;
  color += texture2D(image, uv - offset2).rgb * 0.09447039785044732;
  color += texture2D(image, uv + offset3).rgb * 0.010381362401148057;
  color += texture2D(image, uv - offset3).rgb * 0.010381362401148057;

  return color;
}

void main(void)
{
  vec3 result;

#ifdef MASKED
  float alpha = texture2D(maskTexture, texel).r;

  result.rgb = blur(colorTexture, texel, direction) * alpha * 0.5
      + texture2D(sourceTexture, texel).rgb * (1.0 - alpha);
#else
  result.rgb = blur(colorTexture, texel, direction);
#endif

  color = vec4(result.rgb, 1.0);
}

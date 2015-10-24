#version 410 core

#ifdef AA_SAMPLES
uniform sampler2DMS colorTexture;
#else
uniform sampler2D colorTexture;
#endif

#ifndef AA_SAMPLES
in vec2 texel;
#endif

layout(location = 0) out vec4 color;

void main(void)
{
#ifdef AA_SAMPLES
  ivec2 pos = ivec2(gl_FragCoord.st);
  vec4 output = vec4(0.0);

  for (int i = 0; i < AA_SAMPLES; i++)
    output += texelFetch(colorTexture, pos, i);
  color = output / AA_SAMPLES;
#else
  color = texture2D(colorTexture, texel);
#endif

  color = clamp(color, 0.0, 1.0);
}

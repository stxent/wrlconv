//Fragment Shader

uniform sampler2DRect copper, silk, mask;
uniform vec3 padColor, maskColor, silkColor;
uniform float orientation;

void main(void)
{
  int offset = 1;
  float mult = 0.004;

  vec4 current;
  vec4 pix = texture2DRect(copper, gl_FragCoord.xy);

  float u = 0.5, v = 0.5, h = 1.0;

  int x, y;
  if (texture2DRect(copper, vec2(gl_FragCoord.xy)).a < 0.5)
  {
    for (y = -3; y <= 3; y++)
    {
        if (!y)
        continue;
        for (x = -3; x <= 3; x++)
        {
        if (!x)
            continue;
        current = texture2DRect(copper, vec2(gl_FragCoord.x + x * offset, gl_FragCoord.y + y * offset));
        if (current.a < pix.a)
        {
            u += x * mult;
            v += y * mult * orientation;
        }
        if (current.a > pix.a)
        {
            u -= x * mult;
            v -= y * mult * orientation;
        }
        }
    }
  }
  gl_FragData[0] = vec4(u, v, h, 1.0);
}

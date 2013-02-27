//Fragment Shader

uniform sampler2DRect copper, silk, mask;
uniform vec3 padColor, maskColor, silkColor;

void main(void)
{
  vec4 pixCopper = texture2DRect(copper, gl_FragCoord.xy);
  vec4 pixSilk = texture2DRect(silk, gl_FragCoord.xy);
  vec4 pixMask = texture2DRect(mask, gl_FragCoord.xy);

  if (pixMask.a * pixCopper.a > 0.5)
  {
    gl_FragData[0].rgb = padColor;
  }
  else
  {
    if (pixSilk.a > 0.5)
      gl_FragData[0].rgb = silkColor;
    else
      gl_FragData[0].rgb = maskColor + maskColor * 0.5 * (1 - pixCopper.rgb);
  }
  gl_FragData[0].a = 1.0;
}

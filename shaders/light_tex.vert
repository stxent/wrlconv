// Vertex program
varying vec3 normal, pos, ambientGlobal, ambient, specular;
varying vec4 diffuse;

void main()
{
  normal = normalize(gl_NormalMatrix * gl_Normal);
  pos = vec3(gl_ModelViewMatrix * gl_Vertex);
  //gl_TexCoord[0] = gl_TextureMatrix[0] * gl_MultiTexCoord0;
  gl_TexCoord[0] = gl_MultiTexCoord0;
  /* Compute the diffuse, ambient and globalAmbient terms */
  diffuse = gl_FrontMaterial.diffuse * gl_LightSource[0].diffuse;
  ambient = vec3(gl_FrontMaterial.ambient * gl_LightSource[0].ambient);
  specular = vec3(gl_FrontMaterial.specular * gl_LightSource[0].specular);
  ambientGlobal = vec3(gl_FrontMaterial.ambient * gl_LightModel.ambient);
  gl_Position = ftransform();
}
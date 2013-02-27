// Vertex program
#define LIGHTS 2

varying vec3 normal, pos;
varying vec3 ambientGlobal, ambient[LIGHTS], specular[LIGHTS], diffuse[LIGHTS];

void main()
{
  pos = vec3(gl_ModelViewMatrix * gl_Vertex);
  normal = normalize(gl_NormalMatrix * gl_Normal);
  ambientGlobal = vec3(gl_FrontMaterial.ambient * gl_LightModel.ambient);

  for (int i = 0; i < LIGHTS; i++)
  {
    diffuse[i] = vec3(gl_FrontMaterial.diffuse * gl_LightSource[i].diffuse);
    ambient[i] = vec3(gl_FrontMaterial.ambient * gl_LightSource[i].ambient);
    specular[i] = vec3(gl_FrontMaterial.specular * gl_LightSource[i].specular);
  }

  gl_Position = gl_ModelViewProjectionMatrix * gl_Vertex;
}

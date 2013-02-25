// Vertex program
#define LIGHTS 2
varying vec3 normal, pos, ambientGlobal, ambient[LIGHTS], specular[LIGHTS], diffuse[LIGHTS];
varying vec3 light[LIGHTS];
varying float attenuation[LIGHTS];

void main()
{
  normal = normalize(gl_NormalMatrix * gl_Normal);
  pos = vec3(gl_ModelViewMatrix * gl_Vertex);
  ambientGlobal = vec3(gl_FrontMaterial.ambient * gl_LightModel.ambient);
  gl_TexCoord[0] = gl_MultiTexCoord0;
  for (int i = 0; i < LIGHTS; i++)
  {
    light[i] = -normalize(pos - gl_LightSource[i].position.xyz);
    diffuse[i] = vec3(gl_FrontMaterial.diffuse * gl_LightSource[i].diffuse);
    ambient[i] = vec3(gl_FrontMaterial.ambient * gl_LightSource[i].ambient);
    specular[i] = vec3(gl_FrontMaterial.specular * gl_LightSource[i].specular);
    float len = length(gl_LightSource[i].position.xyz - pos);
    attenuation[i] = 1.0 / (gl_LightSource[i].constantAttenuation + 
                            gl_LightSource[i].linearAttenuation * len + 
                            gl_LightSource[i].quadraticAttenuation * len * len);
  }
  gl_Position = gl_ModelViewProjectionMatrix * gl_Vertex;
}
// Vertex program
#define LIGHTS 2
varying vec3 normal, pos, ambientGlobal, ambient[LIGHTS], specular[LIGHTS], diffuse[LIGHTS];
varying vec3 light[LIGHTS], view;
varying float attenuation[LIGHTS];
attribute vec3 tangent;

void main()
{
  normal = normalize(gl_NormalMatrix * gl_Normal);
  pos = vec3(gl_ModelViewMatrix * gl_Vertex);
  gl_TexCoord[0] = gl_MultiTexCoord0;

  view = normalize(-pos.xyz);
  vec3 tang = normalize(gl_NormalMatrix * tangent);
  vec3 binormal = cross(tang, normal);

  mat3 tbnMatrix = mat3(tang.x, binormal.x, normal.x,
                        tang.y, binormal.y, normal.y,
                        tang.z, binormal.z, normal.z);
  view = tbnMatrix * view;

  ambientGlobal = vec3(gl_FrontMaterial.ambient * gl_LightModel.ambient);
  for (int i = 0; i < LIGHTS; i++)
  {
    diffuse[i] = vec3(gl_FrontMaterial.diffuse * gl_LightSource[i].diffuse);
    ambient[i] = vec3(gl_FrontMaterial.ambient * gl_LightSource[i].ambient);
    specular[i] = vec3(gl_FrontMaterial.specular * gl_LightSource[i].specular);
    light[i] = -normalize(pos - gl_LightSource[i].position.xyz);
    light[i] = tbnMatrix * light[i];
    float len = length(gl_LightSource[i].position.xyz - pos);
    attenuation[i] = 1.0 / (gl_LightSource[i].constantAttenuation + 
                            gl_LightSource[i].linearAttenuation * len + 
                            gl_LightSource[i].quadraticAttenuation * len * len);
  }

  gl_Position = gl_ModelViewProjectionMatrix * gl_Vertex;
}

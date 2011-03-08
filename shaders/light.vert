// Vertex program
varying vec3 normal, pos;
varying vec4 diffuse, ambientGlobal, ambient;

void main() {
  normal = normalize(gl_NormalMatrix * gl_Normal);
  gl_Position = ftransform();
  pos = vec3(gl_ModelViewMatrix * gl_Vertex);
  //gl_TexCoord[0] = gl_TextureMatrix[0] * gl_MultiTexCoord0;
  /* Compute the diffuse, ambient and globalAmbient terms */
  diffuse = gl_FrontMaterial.diffuse * gl_LightSource[0].diffuse;
  ambient = gl_FrontMaterial.ambient * gl_LightSource[0].ambient;
  ambientGlobal = gl_LightModel.ambient * gl_FrontMaterial.ambient;
}
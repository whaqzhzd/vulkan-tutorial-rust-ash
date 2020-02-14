//首先将顶点着色器更改为不再将顶点数据包含在着色器代码本身中
//顶点着色器使用in关键字从顶点缓冲区获取输入 
#version 450
#extension GL_ARB_separate_shader_objects : enable

//顶点属性
//在顶点缓冲区中为每个顶点指定的属性
//https://www.khronos.org/opengl/wiki/Layout_Qualifier_(GLSL)
//layout中location是0和1
//代表vec2消耗1个位置
//color的offest是1
//如果是dvec2则color所在的location是为2
//Scalars and vector types that are not doubles all take up one location. The double and dvec2 types also take one location, while dvec3 and dvec4 take up 2 locations. Structs take up locations based on their member types, in-order. Arrays also take up locations based on their array sizes.
//标量和向量类型不是double都占据一个位置。在双和dvec2类型也需要一个位置，而dvec3和dvec4占用2个位置。结构根据其成员类型按顺序占据位置。阵列还根据其阵列大小占用位置。
layout(location = 0) in vec2 inPosition;
layout(location = 1) in vec3 inColor;

layout(location = 0) out vec3 fragColor;

void main() {
    gl_Position = vec4(inPosition, 0.0, 1.0);
    fragColor = inColor;
}
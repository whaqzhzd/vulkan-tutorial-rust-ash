#version 450
#extension GL_ARB_separate_shader_objects : enable

//修改顶点着色器以包括统一缓冲区对
//请注意uniform，in和out声明的顺序无关紧要。该binding指令类似于location属性的指令。我们将在描述符布局中引用此绑定。
layout(binding = 0) uniform UniformBufferObject {
    mat4 model;
    mat4 view;
    mat4 proj;
} ubo;

layout(location = 0) in vec2 inPosition;
layout(location = 1) in vec3 inColor;

layout(location = 0) out vec3 fragColor;

void main() {
    gl_Position = ubo.proj * ubo.view * ubo.model * vec4(inPosition, 0.0, 1.0);
    fragColor = inColor;
}
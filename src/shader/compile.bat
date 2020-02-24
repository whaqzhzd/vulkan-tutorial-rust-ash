glslc shader.vert -o vert.spv
glslc shader.frag -o frag.spv

glslc vertex_input_shader.vert -o vertex_input_vert.spv
glslc vertex_input_shader.frag -o vertex_input_frag.spv

glslc ubo_shader.vert -o ubo_shader_vert.spv
glslc ubo_shader.frag -o ubo_shader_frag.spv

glslc image_sampler_shader.vert -o image_sampler_vert.spv
glslc image_sampler_shader.frag -o image_sampler_frag.spv

glslc depth_buffering_shader.vert -o depth_buffering_vert.spv
glslc depth_buffering_shader.frag -o depth_buffering_frag.spv

mod totorial;
mod util;

extern crate ash;
extern crate nalgebra as nal;
extern crate winit;

#[macro_use]
extern crate log;
extern crate log4rs;

fn main() {
    let path = std::env::current_dir().ok().unwrap();
    log4rs::init_file(path.join("log4rs.yaml"), Default::default()).unwrap();

    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        panic!("请输入需要运行的示例,例如:cargo run base_code");
    }

    info!("参数列表:{:?}", args);

    #[rustfmt::skip]
    match &*args[1] {
        "base_code"                 | "0"  => totorial::base_code::main(),
        "instance"                  | "1"  => totorial::instance::main(),
        "validation_layers"         | "2"  => totorial::validation_layers::main(),
        "physical_device_selection" | "3"  => totorial::physical_device_selection::main(),
        "logical_device"            | "4"  => totorial::logical_device::main(),
        "window_surface"            | "5"  => totorial::window_surface::main(),
        "swap_chain"                | "6"  => totorial::swap_chain::main(),
        "image_views"               | "7"  => totorial::image_views::main(),
        "graphics_pipeline"         | "8"  => totorial::graphics_pipeline::main(),
        "shader_modules"            | "9"  => totorial::shader_modules::main(),
        "fixed_functions"           | "10" => totorial::fixed_functions::main(),
        "render_passes"             | "11" => totorial::render_passes::main(),
        "graphics_pipeline_complete"| "12" => totorial::graphics_pipeline_complete::main(),
        "framebuffers"              | "13" => totorial::framebuffers::main(),
        "command_buffers"           | "14" => totorial::command_buffers::main(),
        "rendering_and_presentation"| "16" => totorial::rendering_and_presentation::main(),
        "swap_chain_recreation"     | "17" => totorial::swap_chain_recreation::main(),
        "vertex_input_description"  | "18" => totorial::vertex_input_description::main(),
        "vertex_buffer_creation"    | "19" => totorial::vertex_buffer_creation::main(),
        _ => {
            todo!();
        }
    };
}

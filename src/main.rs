mod totorial;
mod util;

extern crate ash;
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
        _ => {
            todo!();
        }
    };
}

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

    match &*args[1] {
        "base_code" => totorial::base_code::main(),
        "instance" => totorial::instance::main(),
        "validation_layers" => totorial::validation_layers::main(),
        _ => {
            todo!();
        }
    };
}

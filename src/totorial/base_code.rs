//!
//!
//! @see https://vulkan-tutorial.com/Drawing_a_triangle/Setup/Base_code
//!
//! 注：本教程所有的英文注释都是有google翻译而来。如有错漏,请告知我修改
//!
//! Note: All English notes in this tutorial are translated from Google. If there are errors and omissions, please let me know
//!
//! The MIT License (MIT)
//!

use winit::{
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::{Window, WindowBuilder},
};

///
/// 最好使用常量而不是硬编码的宽度和高度数字，因为将来我们将多次引用这些值
/// It is better to use constants instead of hard coded width and height numbers, because we will refer to these values more than once in the future
///
const WIDTH: u32 = 800;
///
/// 最好使用常量而不是硬编码的宽度和高度数字，因为将来我们将多次引用这些值
/// It is better to use constants instead of hard coded width and height numbers, because we will refer to these values more than once in the future
///
const HEIGHT: u32 = 600;

struct HelloTriangleApplication;

impl HelloTriangleApplication {
    ///
    /// 初始化窗口
    /// Initialization window
    ///
    /// * `event_loop` 事件循环
    ///
    pub(crate) fn init_window(&mut self, event_loop: &EventLoop<()>) -> Window {
        // 原文采用了glfw来管理窗口
        // The original text uses glfw to manage the window

        // 我决定采用winit
        // I decided to use winit
        WindowBuilder::new()
            .with_title(file!())
            .with_inner_size(winit::dpi::LogicalSize::new(WIDTH, HEIGHT))
            .build(event_loop)
            .expect("[BASE_CODE]:Failed to create window.")
    }

    ///
    ///
    ///
    pub(crate) fn run(&mut self, events: EventLoop<()>) -> Window {
        let window = self.init_window(&events);
        self.init_vulkan();
        self.main_loop(events);
        self.clean_up();

        // 为了在出现错误或窗口关闭之前保持应用程序运行，window必须返回回去,否则出栈的时候windows销毁
        // In order to keep the application running until an error occurs or the window is closed, the window must be returned back, otherwise the windows will be destroyed when it is out of the stack
        window
    }

    ///
    /// 初始化VULKAN
    /// Initialize VULKAN
    ///
    pub(crate) fn init_vulkan(&mut self) {}

    ///
    /// 主循环
    /// Main loop
    ///
    /// 为了在出现错误或窗口关闭之前保持应用程序运行，我们需要向函数添加事件循环
    /// In order to keep the application running until an error occurs or the window closes, we need to add an event loop to the function
    ///
    pub(crate) fn main_loop(&mut self, event_loop: EventLoop<()>) {
        let ptr = self as *mut _;
        event_loop.run(move |event, _, control_flow| {
            *control_flow = ControlFlow::Wait;

            match event {
                Event::LoopDestroyed => unsafe {
                    // winit的实现会直接调用std::process::exit(0);
                    // 这不会调用各种析构函数
                    // 这里我们自己主动调用
                    // pub fn run<F>(mut self, event_handler: F) -> !
                    // where
                    //     F: 'static + FnMut(Event<'_, T>, &RootELW<T>, &mut ControlFlow),
                    // {
                    //     self.run_return(event_handler);
                    //     ::std::process::exit(0);
                    // }
                    std::ptr::drop_in_place(ptr);
                    return;
                },
                Event::WindowEvent { event, .. } => match event {
                    WindowEvent::CloseRequested => {
                        *control_flow = ControlFlow::Exit;
                    }
                    _ => (),
                },
                Event::RedrawRequested(_) => {}
                _ => (),
            }
        });
    }

    ///
    /// 退出清理
    /// Exit cleanup
    ///
    pub(crate) fn clean_up(&mut self) {}
}

///
/// 现在运行该程序时，应会看到一个名为"base_code"的窗口，直到通过关闭该窗口终止应用程序为止。现在，我们已经为 Vulkan 应用程序提供了骨架
/// When you run the program now, you should see a window called "base_code" until you terminate the application by closing the window. Now we have a skeleton for the Vulkan application
///
pub fn main() {
    let events = EventLoop::new();
    let mut hello = HelloTriangleApplication;
    let _win = hello.run(events);
}

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

use ash::{
    version::{EntryV1_0, InstanceV1_0},
    vk::*,
    Entry, Instance,
};
use std::ffi::CString;
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

#[derive(Default)]
struct HelloTriangleApplication {
    ///
    /// vk实例
    ///
    pub instance: Option<Instance>,

    ///
    /// 入口
    ///
    pub entry: Option<Entry>,
}

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
    pub(crate) fn init_vulkan(&mut self) {
        self.instance();
    }

    ///
    /// 主循环
    /// Main loop
    ///
    /// 为了在出现错误或窗口关闭之前保持应用程序运行，我们需要向函数添加事件循环
    /// In order to keep the application running until an error occurs or the window closes, we need to add an event loop to the function
    ///
    pub(crate) fn main_loop(&mut self, event_loop: EventLoop<()>) {
        event_loop.run(move |event, _, control_flow| {
            *control_flow = ControlFlow::Wait;

            match event {
                Event::LoopDestroyed => return,
                Event::WindowEvent { event, .. } => match event {
                    WindowEvent::CloseRequested => {
                        *control_flow = ControlFlow::Exit;
                    }
                    _ => (),
                },
                Event::RedrawRequested(_) => {}
                _ => (),
            }
        })
    }

    ///
    /// 创建vulkan实例
    ///
    pub(crate) fn instance(&mut self) {
        // Creating an instance
        let mut app_info = ApplicationInfo::builder()
            .application_name(CString::new("Hello Triangle").unwrap().as_c_str())
            .engine_name(CString::new("No Engine").unwrap().as_c_str())
            .build();

        let create_info = InstanceCreateInfo::builder()
            .application_info(&app_info)
            .build();

        let entry = Entry::new().unwrap();
        match entry.try_enumerate_instance_version().ok() {
            // Vulkan 1.1+
            Some(version) => {
                let major = ash::vk_version_major!(version.unwrap());
                let minor = ash::vk_version_minor!(version.unwrap());
                let patch = ash::vk_version_patch!(version.unwrap());
                //https://www.khronos.org/registry/vulkan/specs/1.2-extensions/html/vkspec.html#extendingvulkan-coreversions-versionnumbers
                info!("当前支持的VULKAN version_major是:{:?}", major);
                info!("当前支持的VULKAN version_minor是:{:?}", minor);
                info!("当前支持的VULKAN version_patch是:{:?}", patch);

                // Patch version should always be set to 0
                // 引擎版本号
                app_info.engine_version = ash::vk_make_version!(major, minor, 0);
                // 应用名称版本号
                app_info.application_version = ash::vk_make_version!(major, minor, 0);
                // api的版本
                app_info.api_version = ash::vk_make_version!(major, minor, 0);
            }
            // Vulkan 1.0
            None => {
                // 引擎版本号
                app_info.engine_version = ash::vk_make_version!(1, 0, 0);
                // 应用名称版本号
                app_info.application_version = ash::vk_make_version!(1, 0, 0);
                // api的版本
                app_info.api_version = ash::vk_make_version!(1, 0, 0);

                info!("当前支持的VULKAN version_major是:{:?}", 1);
                info!("当前支持的VULKAN version_minor是:{:?}", 0);
                info!("当前支持的VULKAN version_patch是:{:?}", 0);
            }
        }

        // Checking for extension support
        // To retrieve a list of supported extensions before creating an instance, there's the vkEnumerateInstanceExtensionProperties function. It takes a pointer to a variable that stores the number of extensions and an array of VkExtensionProperties to store details of the extensions. It also takes an optional first parameter that allows us to filter extensions by a specific validation layer, which we'll ignore for now.
        // 现在忽略扩展,但我们务必要明确这一点获取扩展的方式 vkEnumerateInstanceExtensionProperties

        // Vulkan 中对象创建函数参数遵循的一般模式是：
        // 使用创建信息进行结构的指针
        // 指向自定义分配器回调的指针
        // 返回创建的对象本事

        // The general pattern for object creation function parameters in Vulkan is:
        // pointer to structure using creation information
        // pointer to custom allocator callback
        // return the created object
        let instance = unsafe {
            entry
                .create_instance(&create_info, None)
                .expect("create_instance error")
        };

        self.instance = Some(instance);
        self.entry = Some(entry);
    }

    ///
    /// 退出清理
    /// Exit cleanup
    ///
    pub(crate) fn clean_up(&mut self) {
        unsafe {
            self.instance.as_ref().unwrap().destroy_instance(None);
        }
    }
}

///
/// 现在运行该程序时，应会看到一个名为"base_code"的窗口，直到通过关闭该窗口终止应用程序为止。现在，我们已经为 Vulkan 应用程序提供了骨架
/// When you run the program now, you should see a window called "base_code" until you terminate the application by closing the window. Now we have a skeleton for the Vulkan application
///
pub fn main() {
    let events = EventLoop::new();
    let mut hello = HelloTriangleApplication::default();
    let _win = hello.run(events);
}

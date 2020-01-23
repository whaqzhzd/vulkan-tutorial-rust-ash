//!
//!
//! @see https://vulkan-tutorial.com/Drawing_a_triangle/Setup/Physical_devices_and_queue_families
//! @see https://www.khronos.org/registry/vulkan/specs/1.1-extensions/html/vkspec.html#VK_EXT_debug_utils
//! cargo run --features=debug window_surface
//!
//! 注：本教程所有的英文注释都是有google翻译而来。如有错漏,请告知我修改
//!
//! Note: All English notes in this tutorial are translated from Google. If there are errors and omissions, please let me know
//!
//! The MIT License (MIT)
//!

use ash::{
    extensions::{ext::DebugUtils, khr::Surface},
    version::{DeviceV1_0, EntryV1_0, InstanceV1_0},
    vk::*,
    Entry, Instance,
};
use std::{
    ffi::{c_void, CStr, CString},
    os::raw::c_char,
};
use winit::{
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::{Window, WindowBuilder},
};

#[cfg(target_os = "windows")]
use ash::extensions::khr::Win32Surface;

///
/// VK_LAYER_KHRONOS_validation 是标准验证的绑定层
///
/// 请注意这个修改
/// 如果你的vk版本过低
/// 使用VK_LAYER_KHRONOS_validation将会报错
/// @ https://vulkan.lunarg.com/doc/view/1.1.108.0/mac/validation_layers.html
///
const VALIDATION_LAYERS: [&'static str; 1] = ["VK_LAYER_KHRONOS_validation"; 1];

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

///
/// 支持绘图命令的队列族和支持显示的队列族可能不会重叠
///
#[derive(Default)]
struct QueueFamilyIndices {
    ///
    /// 图形命令队列族
    ///
    pub graphics_family: Option<u32>,

    ///
    /// 显示命令队列族
    ///
    pub present_family: Option<u32>,
}

impl QueueFamilyIndices {
    pub fn is_complete(&self) -> bool {
        self.graphics_family.is_some() && self.present_family.is_some()
    }
}

#[derive(Default)]
struct HelloTriangleApplication {
    ///
    /// 窗口
    ///
    pub(crate) win: Option<winit::window::Window>,

    ///
    /// vk实例
    ///
    pub(crate) instance: Option<Instance>,

    ///
    /// 入口
    ///
    pub(crate) entry: Option<Entry>,

    ///
    /// 调试信息
    ///
    pub(crate) debug_messenger: Option<DebugUtilsMessengerEXT>,

    ///
    /// 调试
    ///
    pub(crate) debug_utils_loader: Option<DebugUtils>,

    ///
    /// 本机可使用的物理设备
    ///
    pub(crate) physical_devices: Vec<PhysicalDevice>,

    ///
    /// 选中的可用的物理设备
    ///
    pub(crate) physical_device: Option<PhysicalDevice>,

    ///
    /// 逻辑设备
    ///
    pub(crate) device: Option<ash::Device>,

    ///
    /// 存储图形队列的句柄
    ///
    pub(crate) graphics_queue: Queue,

    ///
    /// 存储显示队列的句柄
    ///
    pub(crate) present_queue: Queue,

    ///
    /// 表面抽象加载器
    ///
    pub(crate) surface_loader: Option<Surface>,

    ///
    /// 由于Vulkan是与平台无关的API，因此它无法直接直接与窗口系统交互。为了在Vulkan和窗口系统之间建立连接以将结果呈现给屏幕，我们需要使用WSI（窗口系统集成）扩展
    ///
    pub(crate) surface: Option<SurfaceKHR>,
}

unsafe extern "system" fn debug_callback(
    message_severity: DebugUtilsMessageSeverityFlagsEXT,
    message_types: DebugUtilsMessageTypeFlagsEXT,
    p_callback_data: *const DebugUtilsMessengerCallbackDataEXT,
    _p_user_data: *mut c_void,
) -> u32 {
    // 此枚举的值设置方式，可以使用比较操作来检查消息与某些严重性级别相比是否相等或更糟
    // use std::cmp::Ordering;
    //
    // if message_severity.cmp(&DebugUtilsMessageSeverityFlagsEXT::WARNING) > Ordering::Greater {
    //
    // }

    match message_severity {
        DebugUtilsMessageSeverityFlagsEXT::VERBOSE => {
            info!("debug_callback message_severity VERBOSE")
        }
        DebugUtilsMessageSeverityFlagsEXT::WARNING => {
            info!("debug_callback message_severity WARNING")
        }
        DebugUtilsMessageSeverityFlagsEXT::ERROR => info!("debug_callback message_severity ERROR"),
        DebugUtilsMessageSeverityFlagsEXT::INFO => info!("debug_callback message_severity INFO"),
        _ => info!("debug_callback message_severity DEFAULT"),
    };

    match message_types {
        DebugUtilsMessageTypeFlagsEXT::GENERAL => info!("debug_callback message_types GENERAL"),
        DebugUtilsMessageTypeFlagsEXT::PERFORMANCE => {
            info!("debug_callback message_types PERFORMANCE")
        }
        DebugUtilsMessageTypeFlagsEXT::VALIDATION => {
            info!("debug_callback message_types VALIDATION")
        }
        _ => info!("debug_callback message_types DEFAULT"),
    };

    info!(
        "debug_callback : {:?}",
        CStr::from_ptr((*p_callback_data).p_message)
    );

    FALSE
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
    pub(crate) fn run(&mut self, events: &EventLoop<()>) -> () {
        self.win = Some(self.init_window(events));
        self.init_vulkan();
    }

    ///
    /// 初始化VULKAN
    /// Initialize VULKAN
    ///
    pub(crate) fn init_vulkan(&mut self) {
        self.instance();
        self.setup_debug_messenger();
        self.create_surface();
        self.pick_physical_device();
        self.create_logical_device();
    }

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
                Event::LoopDestroyed => return,
                Event::WindowEvent { event, .. } => match event {
                    WindowEvent::CloseRequested => {
                        *control_flow = ControlFlow::Exit;

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
                        unsafe {
                            std::ptr::drop_in_place(ptr);
                        }
                    }
                    _ => (),
                },
                Event::RedrawRequested(_) => {}
                _ => (),
            }
        });
    }

    ///
    /// 创建vulkan实例
    ///
    pub(crate) fn instance(&mut self) {
        let entry = Entry::new().unwrap();
        self.entry = Some(entry);

        // 首先校验我们需要启用的层当前vulkan扩展是否支持
        // First check if the layer we need to enable currently vulkan extension supports
        //
        if cfg!(feature = "debug") && !self.check_validation_layer_support() {
            panic!("validation layers requested, but not available! @see https://vulkan.lunarg.com/doc/view/1.1.108.0/mac/validation_layers.html");
        };

        // Creating an instance
        let mut app_info = ApplicationInfo::builder()
            .application_name(CString::new("Hello Triangle").unwrap().as_c_str())
            .engine_name(CString::new("No Engine").unwrap().as_c_str())
            .build();

        let extensions = self.get_required_extensions();
        let mut create_info = InstanceCreateInfo::builder()
            .application_info(&app_info)
            .enabled_extension_names(&extensions)
            .build();

        let cstr_argv: Vec<_> = VALIDATION_LAYERS
            .iter()
            .map(|arg| CString::new(*arg).unwrap())
            .collect();
        let p_argv: Vec<_> = cstr_argv.iter().map(|arg| arg.as_ptr()).collect();
        if cfg!(feature = "debug") {
            let debug_utils_create_info = Self::populate_debug_messenger_create_info();
            create_info.enabled_layer_count = p_argv.len() as u32;
            create_info.pp_enabled_layer_names = p_argv.as_ptr();
            create_info.p_next = &debug_utils_create_info as *const DebugUtilsMessengerCreateInfoEXT
                as *const c_void;
        };

        match self
            .entry
            .as_ref()
            .unwrap()
            .try_enumerate_instance_version()
            .ok()
        {
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
            self.entry
                .as_ref()
                .unwrap()
                .create_instance(&create_info, None)
                .expect("create_instance error")
        };

        self.instance = Some(instance);
    }

    ///
    /// 创建窗口表面
    ///
    pub(crate) fn create_surface(&mut self) {
        self.surface_loader = Some(Surface::new(
            self.entry.as_ref().unwrap(),
            self.instance.as_ref().unwrap(),
        ));
        // 根据不同平台创建窗口表面
        if cfg!(target_os = "windows") {
            use winapi::{shared::windef::HWND, um::libloaderapi::GetModuleHandleW};
            use winit::platform::windows::WindowExtWindows;
            // 构建KHR表面创建信息结构实例
            let mut create_info = Win32SurfaceCreateInfoKHR::default();
            let hinstance = unsafe { GetModuleHandleW(std::ptr::null()) as *const c_void };
            create_info.hinstance = hinstance;
            // 给定类型
            create_info.s_type = StructureType::WIN32_SURFACE_CREATE_INFO_KHR;
            // 自定义数据指针
            create_info.p_next = std::ptr::null();
            // 使用的标志
            create_info.flags = Win32SurfaceCreateFlagsKHR::all();
            // 窗体句柄
            let hwnd = self.win.as_ref().unwrap().hwnd() as HWND;
            create_info.hwnd = hwnd as *const c_void;

            let win32_surface_loader = Win32Surface::new(
                self.entry.as_ref().unwrap(),
                self.instance.as_ref().unwrap(),
            );
            self.surface = unsafe {
                Some(
                    win32_surface_loader
                        .create_win32_surface(&create_info, None)
                        .expect("create_swapchain error"),
                )
            };
        }

        if cfg!(target_os = "android") {
            todo!();
        }

        if cfg!(target_os = "ios") {
            todo!();
        }
    }

    ///
    ///
    ///
    pub(crate) fn setup_debug_messenger(&mut self) {
        let debug_utils_create_info = Self::populate_debug_messenger_create_info();

        let debug_utils_loader: DebugUtils = DebugUtils::new(
            self.entry.as_ref().unwrap(),
            self.instance.as_ref().unwrap(),
        );

        // https://www.khronos.org/registry/vulkan/specs/1.1-extensions/html/vkspec.html#VK_EXT_debug_utils
        self.debug_messenger = unsafe {
            Some(
                debug_utils_loader
                    .create_debug_utils_messenger(&debug_utils_create_info, None)
                    .expect("failed to set up debug messenger!"),
            )
        };

        self.debug_utils_loader = Some(debug_utils_loader);
    }

    ///
    /// 检查可用的物理设备
    ///
    pub(crate) fn pick_physical_device(&mut self) {
        let physical_devices: Vec<PhysicalDevice> = unsafe {
            self.instance
                .as_ref()
                .unwrap()
                .enumerate_physical_devices()
                .expect("enumerate_physical_devices error")
        };

        if physical_devices.len() <= 0 {
            panic!("failed to find GPUs with Vulkan support!");
        }

        for physical_device in physical_devices.iter() {
            if self.is_device_suitable(physical_device) {
                self.physical_device = Some(*physical_device);
            }
        }
        self.physical_devices = physical_devices;

        if let None::<PhysicalDevice> = self.physical_device {
            panic!("failed to find a suitable GPU!");
        }
    }

    ///
    /// 创建逻辑设备
    ///
    pub(crate) fn create_logical_device(&mut self) {
        let physical_device = self.physical_device.as_ref().unwrap();
        let indices = self.find_queue_families(physical_device);

        let mut queue_create_infos = Vec::<DeviceQueueCreateInfo>::new();
        let unique_queue_families = vec![
            indices.graphics_family.unwrap(),
            indices.present_family.unwrap(),
        ];

        // https://vulkan.lunarg.com/doc/view/1.1.130.0/windows/chunked_spec/chap4.html#devsandqueues-priority
        //较高的值表示较高的优先级，其中0.0是最低优先级，而1.0是最高优先级。
        let queue_priority = 1.0f32;
        for i in unique_queue_families.iter() {
            //https://vulkan.lunarg.com/doc/view/1.1.130.0/windows/chunked_spec/chap4.html#VkDeviceQueueCreateInfo

            // 此结构描述了单个队列族所需的队列数
            // This structure describes the number of queues we want for a single queue family.

            //当前可用的驱动程序将只允许您为每个队列系列创建少量队列，而您实际上并不需要多个。这是因为您可以在多个线程上创建所有命令缓冲区，然后通过一次低开销调用在主线程上全部提交。
            //The currently available drivers will only allow you to create a small number of queues for each queue family and you don't really need more than one. That's because you can create all of the command buffers on multiple threads and then submit them all at once on the main thread with a single low-overhead call.
            let queue_create_info = DeviceQueueCreateInfo::builder()
                .queue_family_index(*i)
                .queue_priorities(&[queue_priority])
                .build();

            queue_create_infos.push(queue_create_info);
        }

        //指定使用的设备功能
        let device_features = PhysicalDeviceFeatures::default();
        //创建逻辑设备
        let mut device_create_info = DeviceCreateInfo::builder()
            .queue_create_infos(&queue_create_infos)
            .enabled_features(&device_features)
            .build();
        device_create_info.enabled_extension_count = 0;
        // 兼容实现部分暂不实现
        // if (enableValidationLayers) {
        //     createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
        //     createInfo.ppEnabledLayerNames = validationLayers.data();
        // }
        device_create_info.enabled_layer_count = 0;

        // 实例化逻辑设备
        self.device = unsafe {
            Some(
                self.instance
                    .as_ref()
                    .unwrap()
                    .create_device(self.physical_device.unwrap(), &device_create_info, None)
                    .expect("Failed to create logical device!"),
            )
        };

        //检索队列句柄
        //队列是与逻辑设备一起自动创建的，但是我们尚无与之交互的句柄
        //由于我们仅从该队列族创建单个队列，因此我们将仅使用index 0。

        //队列族索引相同，则两个句柄现在很可能具有相同的值
        self.graphics_queue = unsafe {
            self.device
                .as_ref()
                .unwrap()
                .get_device_queue(indices.graphics_family.unwrap(), 0)
        };

        self.present_queue = unsafe {
            self.device
                .as_ref()
                .unwrap()
                .get_device_queue(indices.present_family.unwrap(), 0)
        };
    }

    pub(crate) fn is_device_suitable(&mut self, device: &PhysicalDevice) -> bool {
        let indices = self.find_queue_families(device);

        return indices.is_complete();
    }

    pub(crate) fn find_queue_families(&self, device: &PhysicalDevice) -> QueueFamilyIndices {
        let mut indices = QueueFamilyIndices::default();

        let physical_device_properties: PhysicalDeviceProperties = unsafe {
            self.instance
                .as_ref()
                .unwrap()
                .get_physical_device_properties(*device)
        };

        info!(
            "物理设备名称: {:?}",
            Self::char2str(&physical_device_properties.device_name)
        );
        info!("物理设备类型: {:?}", physical_device_properties.device_type);

        //https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VkPhysicalDeviceLimits.html
        info!("物理设备属性：{:#?}", physical_device_properties);

        let physical_device_features: PhysicalDeviceFeatures = unsafe {
            self.instance
                .as_ref()
                .unwrap()
                .get_physical_device_features(*device)
        };
        info!(
            "物理设备是否支持几何着色器: {:?}",
            physical_device_features.geometry_shader
        );

        let queue_families: Vec<QueueFamilyProperties> = unsafe {
            self.instance
                .as_ref()
                .unwrap()
                .get_physical_device_queue_family_properties(*device)
        };

        for (i, queue_familie) in queue_families.iter().enumerate() {
            // 必须支持图形队列
            if queue_familie.queue_flags.contains(QueueFlags::GRAPHICS) {
                indices.graphics_family = Some(i as u32);
            }

            let is_present_support = unsafe {
                self.surface_loader
                    .as_ref()
                    .unwrap()
                    .get_physical_device_surface_support(*device, i as u32, self.surface.unwrap())
            };

            // 又必须支持显示队列
            if is_present_support {
                indices.present_family = Some(i as u32);
            }

            if indices.is_complete() {
                break;
            }
        }

        indices
    }

    pub(crate) fn populate_debug_messenger_create_info() -> DebugUtilsMessengerCreateInfoEXT {
        DebugUtilsMessengerCreateInfoEXT::builder()
            .message_severity(
                DebugUtilsMessageSeverityFlagsEXT::VERBOSE
                    | DebugUtilsMessageSeverityFlagsEXT::WARNING
                    | DebugUtilsMessageSeverityFlagsEXT::ERROR,
            )
            .message_type(
                DebugUtilsMessageTypeFlagsEXT::GENERAL
                    | DebugUtilsMessageTypeFlagsEXT::VALIDATION
                    | DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
            )
            .pfn_user_callback(Some(debug_callback))
            .build()
    }

    ///
    /// 退出清理
    /// Exit cleanup
    ///
    pub(crate) fn clean_up(&mut self) {
        unsafe {
            info!("clean_up");

            // 如果不销毁debug_messenger而直接销毁instance
            // 则会发出如下警告:
            // debug_callback : "OBJ ERROR : For VkInstance 0x1db513db8a0[], VkDebugUtilsMessengerEXT 0x2aefa40000000001[] has not been destroyed. The Vulkan spec states: All child objects created using instance must have been destroyed prior to destroying instance (https://www.khronos.org/registry/vulkan/specs/1.1-extensions/html/vkspec.html#VUID-vkDestroyInstance-instance-00629)"
            if cfg!(feature = "debug") {
                if let Some(debug_messenger) = self.debug_messenger {
                    self.debug_utils_loader
                        .as_ref()
                        .unwrap()
                        .destroy_debug_utils_messenger(debug_messenger, None);
                }
            }

            if let Some(instance) = self.instance.as_ref() {
                if let Some(surface_khr) = self.surface {
                    self.surface_loader
                        .as_ref()
                        .unwrap()
                        .destroy_surface(surface_khr, None);
                }

                if let Some(device) = self.device.as_ref() {
                    device.destroy_device(None);
                }

                instance.destroy_instance(None);
            }
        }
    }

    ///
    /// 请求扩展
    ///
    /// glfwGetRequiredInstanceExtensions
    ///
    ///
    pub(crate) fn get_required_extensions(&mut self) -> Vec<*const i8> {
        let mut v = Vec::new();
        if cfg!(target_os = "windows") {
            v.push(Surface::name().as_ptr());
            v.push(Win32Surface::name().as_ptr());
            v.push(DebugUtils::name().as_ptr());
        };

        if cfg!(target_os = "macos") {
            todo!();
        };

        if cfg!(target_os = "android") {
            todo!();
        }

        v
    }

    ///
    /// 校验需要启用的层当前vulkan实力层是否支持
    /// Verify that the layer that needs to be enabled currently supports the vulkan strength layer
    ///
    pub(crate) fn check_validation_layer_support(&mut self) -> bool {
        // 获取总的验证Layer信息
        let layer_properties: Vec<LayerProperties> = self
            .entry
            .as_ref()
            .unwrap()
            .enumerate_instance_layer_properties()
            .expect("Failed to enumerate instance layers properties");

        info!("layer_properties{:?}", layer_properties);

        for layer_name in VALIDATION_LAYERS.iter() {
            let mut layer_found = false;

            for layer_propertie in layer_properties.iter() {
                if Self::char2str(&layer_propertie.layer_name) == layer_name.to_string() {
                    layer_found = true;
                }
            }

            if !layer_found {
                return false;
            }
        }

        true
    }

    pub(crate) fn char2str(char: &[c_char]) -> String {
        let raw_string = unsafe {
            let pointer = char.as_ptr();
            CStr::from_ptr(pointer)
        };

        raw_string
            .to_str()
            .expect("Failed to convert vulkan raw string.")
            .to_string()
    }
}

impl Drop for HelloTriangleApplication {
    fn drop(&mut self) {
        self.clean_up();
    }
}

///
/// Vulkan API 是围绕最小驱动程序开销的想法设计的，该目标的表现之一是默认情况下 API 中的错误检查非常有限。即使
/// 像将枚举设置为不正确的值或将空指针传递给所需参数等简单错误，通常也不会显式处理，只会导致崩溃或未定义的行为。由于
/// Vulkan 要求您非常明确地了解所做的一切，因此很容易犯许多小错误，例如使用新的 GPU 功能，并忘记在逻辑设备创建时请求它。
/// 但是，这并不意味着无法将这些检查添加到 API 中。Vulkan为此引入了一个优雅的系统，称为验证层.
/// 验证层是连接到 Vulkan 函数调用以应用其他操作的可选组件
///
/// 验证层的常见操作包括:
/// 对照规范检查参数值以检测误用
/// 跟踪对象的创建和销毁，以查找资源泄漏
/// 通过跟踪来自调用的线程来检查线程安全
/// 将每个调用及其参数记录到标准输出
/// 跟踪Vulkan需要分析并重播
///
pub fn main() {
    let events = EventLoop::new();
    let mut hello = HelloTriangleApplication::default();
    hello.run(&events);
    hello.main_loop(events);
}

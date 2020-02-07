//!
//!
//! @see https://vulkan-tutorial.com/Drawing_a_triangle/Setup/Physical_devices_and_queue_families
//! @see https://www.khronos.org/registry/vulkan/specs/1.1-extensions/html/vkspec.html#VK_EXT_debug_utils
//! cargo run --features=debug command_buffers
//!
//! 注：本教程所有的英文注释都是有google翻译而来。如有错漏,请告知我修改
//!
//! Note: All English notes in this tutorial are translated from Google. If there are errors and omissions, please let me know
//!
//! The MIT License (MIT)
//!
use ash::{
    extensions::{
        ext::DebugUtils,
        khr::{Surface, Swapchain},
    },
    version::{DeviceV1_0, EntryV1_0, InstanceV1_0},
    vk::*,
    Entry, Instance,
};
use std::{
    ffi::{c_void, CStr, CString},
    io::Cursor,
    os::raw::c_char,
    path::Path,
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
/// 由于图像表示与窗口系统以及与窗口相关的表面紧密相关，因此它实际上不是Vulkan核心的一部分。
/// 启用扩展VK_KHR_swapchain
///
const DEVICE_EXTENSIONES: [&'static str; 1] = ["VK_KHR_swapchain"; 1];

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

///
/// 查询到的交换链支持的详细信息
///
#[derive(Default)]
struct SwapChainSupportDetails {
    ///
    /// 基本表面功能（交换链中图像的最小/最大数量，图像的最小/最大宽度和高度）
    ///
    pub capabilities: SurfaceCapabilitiesKHR,

    ///
    /// 表面格式（像素格式，色彩空间）
    ///
    pub formats: Vec<SurfaceFormatKHR>,

    ///
    /// 可用的显示模式
    ///
    pub present_modes: Vec<PresentModeKHR>,
}

#[derive(Default)]
struct HelloTriangleApplication {
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

    ///
    /// 交换链加载器
    ///
    pub(crate) swap_chain_loader: Option<Swapchain>,

    ///
    /// 交换链对象
    ///
    pub(crate) swap_chain: SwapchainKHR,

    ///
    /// 存储句柄
    ///
    pub(crate) swap_chain_images: Vec<Image>,

    ///
    /// 交换链格式化类型
    ///
    pub(crate) swap_chain_image_format: Format,

    ///
    /// 交换链大小
    ///
    pub(crate) swap_chain_extent: Extent2D,

    ///
    /// 交换链视图操作
    ///
    pub(crate) swap_chain_image_views: Vec<ImageView>,

    ///
    /// 渲染通道
    ///
    pub(crate) render_pass: RenderPass,

    ///
    /// 管道布局
    ///
    pub(crate) pipeline_layout: PipelineLayout,

    ///
    /// 图形管线
    ///
    pub(crate) graphics_pipeline: Pipeline,

    ///
    /// 帧缓冲
    ///
    pub(crate) swap_chain_framebuffers: Vec<Framebuffer>,

    ///
    /// 创建命令池，然后才能创建命令缓冲区。命令池管理用于存储缓冲区的内存，并从中分配命令缓冲区
    ///
    pub(crate) command_pool: CommandPool,

    ///
    ///
    ///
    pub(crate) command_buffers: Vec<CommandBuffer>,
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
    pub(crate) fn init_window(&mut self, event_loop: &EventLoop<()>) -> winit::window::Window {
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
    pub(crate) fn run(&mut self, events: &EventLoop<()>) -> winit::window::Window {
        let win = self.init_window(events);
        self.init_vulkan(&win);

        win
    }

    ///
    /// 初始化VULKAN
    /// Initialize VULKAN
    ///
    pub(crate) fn init_vulkan(&mut self, win: &winit::window::Window) {
        self.instance();
        self.setup_debug_messenger();
        self.create_surface(win);
        self.pick_physical_device();
        self.create_logical_device();
        self.create_swap_chain();
        self.create_image_views();
        self.create_render_pass();
        self.create_graphics_pipeline();
        self.create_framebuffers();
        self.create_command_pool();
        self.create_command_buffers();
    }

    ///
    /// 主循环
    /// Main loop
    ///
    /// 为了在出现错误或窗口关闭之前保持应用程序运行，我们需要向函数添加事件循环
    /// In order to keep the application running until an error occurs or the window closes, we need to add an event loop to the function
    ///
    pub(crate) fn main_loop(&mut self, event_loop: EventLoop<()>, window: Window) {
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
                Event::MainEventsCleared => {}
                Event::RedrawRequested(_) => {
                    window.request_redraw();
                }
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

        let debug_utils_create_info = Self::populate_debug_messenger_create_info();
        if cfg!(feature = "debug") {
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
    pub(crate) fn create_surface(&mut self, win: &winit::window::Window) {
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
            let hwnd = win.hwnd() as HWND;
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

        //@see https://www.reddit.com/r/vulkan/comments/7rt0o1/questions_about_queue_family_indices_and_high_cpu/
        let mut unique_queue_families: Vec<u32> = Vec::<u32>::new();
        if indices.graphics_family.unwrap() == indices.present_family.unwrap() {
            unique_queue_families.push(indices.graphics_family.unwrap());
        } else {
            unique_queue_families.push(indices.graphics_family.unwrap());
            unique_queue_families.push(indices.present_family.unwrap());
        };

        // https://vulkan.lunarg.com/doc/view/1.1.130.0/windows/chunked_spec/chap4.html#devsandqueues-priority
        //较高的值表示较高的优先级，其中0.0是最低优先级，而1.0是最高优先级。
        let queue_priority = 1.0f32;
        for (_i, &unique_queue_familie) in unique_queue_families.iter().enumerate() {
            //https://vulkan.lunarg.com/doc/view/1.1.130.0/windows/chunked_spec/chap4.html#VkDeviceQueueCreateInfo

            // 此结构描述了单个队列族所需的队列数
            // This structure describes the number of queues we want for a single queue family.

            //当前可用的驱动程序将只允许您为每个队列系列创建少量队列，而您实际上并不需要多个。这是因为您可以在多个线程上创建所有命令缓冲区，然后通过一次低开销调用在主线程上全部提交。
            //The currently available drivers will only allow you to create a small number of queues for each queue family and you don't really need more than one. That's because you can create all of the command buffers on multiple threads and then submit them all at once on the main thread with a single low-overhead call.
            let queue_create_info = DeviceQueueCreateInfo::builder()
                .queue_family_index(unique_queue_familie)
                .queue_priorities(&[queue_priority])
                .build();

            queue_create_infos.push(queue_create_info);
        }

        //指定使用的设备功能
        let device_features = PhysicalDeviceFeatures::default();

        //现在需要启用交换链
        let cstr_exts: Vec<_> = DEVICE_EXTENSIONES
            .iter()
            .map(|arg| CString::new(*arg).unwrap())
            .collect();
        let csstr_exts: Vec<_> = cstr_exts.iter().map(|arg| arg.as_ptr()).collect();

        //创建逻辑设备
        let mut device_create_info = DeviceCreateInfo::builder()
            .queue_create_infos(&queue_create_infos)
            .enabled_features(&device_features)
            .enabled_extension_names(&csstr_exts)
            .build();

        //现在启用的扩展数量由enabled_extension_names方法设置
        //device_create_info.enabled_extension_count = 0;

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

    ///
    /// 创建交换链
    ///
    pub(crate) fn create_swap_chain(&mut self) {
        let swap_chain_support =
            self.query_swap_chain_support(self.physical_device.as_ref().unwrap());

        let surface_format = self.choose_swap_surface_format(swap_chain_support.formats);
        let present_mode = self.choose_swap_present_mode(swap_chain_support.present_modes);
        let extent = self.choose_swap_extent(&swap_chain_support.capabilities);

        //除了这些属性外，我们还必须确定交换链中要包含多少个图像。该实现指定其运行所需的最小数量：
        //仅坚持最低限度意味着我们有时可能需要等待驱动程序完成内部操作，然后才能获取要渲染的一张图像。因此，建议您至少请求至少一张图片
        let mut image_count = swap_chain_support.capabilities.min_image_count + 1;
        //还应确保不超过最大图像数

        if swap_chain_support.capabilities.max_image_count > 0
            && image_count > swap_chain_support.capabilities.max_image_count
        {
            image_count = swap_chain_support.capabilities.max_image_count;
        }

        let mut create_info = SwapchainCreateInfoKHR::builder()
            .surface(self.surface.unwrap())
            .min_image_count(image_count)
            .image_format(surface_format.format)
            .image_color_space(surface_format.color_space)
            //Try removing the createInfo.imageExtent = extent; line with validation layers enabled. You'll see that one of the validation layers immediately catches the mistake and a helpful message is printed:
            .image_extent(extent)
            //imageArrayLayers指定层的每个图像包括的量
            //除非您正在开发立体3D应用程序，否则始终如此为1
            .image_array_layers(1)
            .image_usage(ImageUsageFlags::COLOR_ATTACHMENT)
            .build();

        let physical_device = self.physical_device.as_ref().unwrap();
        let indices = self.find_queue_families(physical_device);
        let queue_familie_indices = vec![
            indices.graphics_family.unwrap(),
            indices.present_family.unwrap(),
        ];

        if indices.graphics_family != indices.present_family {
            //VK_SHARING_MODE_CONCURRENT：图像可以在多个队列族中使用，而无需明确的所有权转移。
            create_info.image_sharing_mode = SharingMode::CONCURRENT;
            create_info.queue_family_index_count = 2;
            create_info.p_queue_family_indices = queue_familie_indices.as_ptr();
        } else {
            //如果队列族不同，那么在本教程中我们将使用并发模式以避免执行所有权
            //VK_SHARING_MODE_EXCLUSIVE：图像一次由一个队列族拥有，并且必须在其他队列族中使用图像之前显式转移所有权。此选项提供最佳性能。
            create_info.image_sharing_mode = SharingMode::EXCLUSIVE;
            create_info.queue_family_index_count = 0;
            create_info.p_queue_family_indices = std::ptr::null();
        }
        //我们可以指定某一变换应适用于在交换链图像
        //要指定您不希望进行任何转换，只需指定当前转换即可。
        create_info.pre_transform = swap_chain_support.capabilities.current_transform;
        //指定是否应将Alpha通道用于与窗口系统中的其他窗口混合
        create_info.composite_alpha = CompositeAlphaFlagsKHR::OPAQUE;
        create_info.present_mode = present_mode;
        // 设置为true，意味着我们不在乎被遮盖的像素的颜色
        // 除非您真的需要能够读回这些像素并获得可预测的结果，否则通过启用裁剪将获得最佳性能。
        create_info.clipped = TRUE;
        //剩下最后一个场oldSwapChain。使用Vulkan时，您的交换链可能在应用程序运行时无效或未优化，例如，因为调整了窗口大小。在这种情况下，实际上需要从头开始重新创建交换链，并且必须在该字段中指定对旧交换链的引用。这是一个复杂的主题，我们将在以后的章节中了解更多。现在，我们假设我们只会创建一个交换链。
        create_info.old_swapchain = SwapchainKHR::null();

        let swapchain_loader = Swapchain::new(
            self.instance.as_ref().unwrap(),
            self.device.as_ref().unwrap(),
        );

        //创建交换链
        self.swap_chain = unsafe {
            swapchain_loader
                .create_swapchain(&create_info, None)
                .expect("create_swapchain error")
        };

        self.swap_chain_images = unsafe {
            swapchain_loader
                .get_swapchain_images(self.swap_chain)
                .expect("Failed to get Swapchain Images.")
        };

        info!("交换链数量:{:?}", self.swap_chain_images.len());

        self.swap_chain_image_format = surface_format.format;
        self.swap_chain_extent = extent;
        self.swap_chain_loader = Some(swapchain_loader);
    }

    ///
    /// 创建视图
    ///
    pub(crate) fn create_image_views(&mut self) {
        let len = self.swap_chain_images.len();
        self.swap_chain_image_views.reserve(len);

        for i in 0..len {
            let info = ImageViewCreateInfo::builder()
                .image(self.swap_chain_images[i])
                //该viewType参数允许您将图像视为1D纹理，2D纹理，3D纹理和立方体贴图。
                .view_type(ImageViewType::TYPE_2D)
                .format(self.swap_chain_image_format)
                //The components field allows you to swizzle the color channels around. For example, you can map all of the channels to the red channel for a monochrome texture. You can also map constant values of 0 and 1 to a channel. In our case we'll stick to the default mapping.
                .components(ComponentMapping {
                    r: ComponentSwizzle::IDENTITY,
                    g: ComponentSwizzle::IDENTITY,
                    b: ComponentSwizzle::IDENTITY,
                    a: ComponentSwizzle::IDENTITY,
                })
                //The subresourceRange field describes what the image's purpose is and which part of the image should be accessed. Our images will be used as color targets without any mipmapping levels or multiple layers.
                .subresource_range(ImageSubresourceRange {
                    aspect_mask: ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                })
                .build();

            self.swap_chain_image_views.insert(i, unsafe {
                self.device
                    .as_ref()
                    .unwrap()
                    .create_image_view(&info, None)
                    .expect("create_image_view error")
            });
        }
    }

    ///
    ///创建渲染过程
    ///
    pub(crate) fn create_render_pass(&mut self) {
        //当前示例只有一个颜色缓冲区附件，该附件由交换链中的图像之一
        let color_attachment = AttachmentDescription::builder()
            //匹配交换链图像的格式
            .format(self.swap_chain_image_format)
            .samples(SampleCountFlags::TYPE_1)
            //loadOp与storeOp适用于颜色和深度数据，和stencilLoadOp/ stencilStoreOp适用于模板数据
            //ATTACHMENT_LOAD_OP_LOAD 保留附件的现有内容
            //ATTACHMENT_LOAD_OP_CLEAR 在开始时将值清除为常数
            //ATTACHMENT_LOAD_OP_DONT_CARE 现有内容未定义；我们不在乎他们
            .load_op(AttachmentLoadOp::CLEAR)
            //ATTACHMENT_STORE_OP_STORE 渲染的内容将存储在内存中，以后可以读取
            //ATTACHMENT_STORE_OP_DONT_CARE 渲染操作后，帧缓冲区的内容将不确定
            .store_op(AttachmentStoreOp::STORE)
            //适用于模板数据
            //我们的应用程序不会对模板缓冲区做任何事情，因此加载和存储的结果无关紧要。
            .stencil_load_op(AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(AttachmentStoreOp::DONT_CARE)
            //Vulkan中的纹理和帧缓冲区由VkImage具有特定像素格式的对象表示，但是内存中像素的布局可以根据您要对图像进行的处理而更改。
            //IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL：用作颜色附件的图像
            //IMAGE_LAYOUT_PRESENT_SRC_KHR：要在交换链中显示的图像
            //IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL：用作存储器复制操作目的地的图像
            .initial_layout(ImageLayout::UNDEFINED)
            .final_layout(ImageLayout::PRESENT_SRC_KHR)
            .build();

        //子通道和附件
        //单个渲染过程可以包含多个子过程。
        //例如，一系列后处理效果依次应用。如果将这些渲染操作分组到一个渲染通道中，则Vulkan能够对这些操作进行重新排序并节省内存带宽，以实现更好的性能。但是，对于第一个三角形，我们将坚持使用单个子通道。
        let color_attachment_ref = AttachmentReference::builder()
            .layout(ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .build();

        //子通道可以引用以下其他类型的附件
        //pInputAttachments：从着色器读取的附件
        //pResolveAttachments：用于多采样颜色附件的附件
        //pDepthStencilAttachment：深度和模板数据的附件
        //pPreserveAttachments：此子通道未使用的附件，但必须保留其数据
        let sub_pass = SubpassDescription::builder()
            //Vulkan将来可能还会支持计算子通道，因此我们必须明确地说这是图形子通道
            .pipeline_bind_point(PipelineBindPoint::GRAPHICS)
            .color_attachments(&[color_attachment_ref])
            .build();

        //渲染通道
        let render_pass_info = RenderPassCreateInfo::builder()
            .attachments(&[color_attachment])
            .subpasses(&[sub_pass])
            .build();

        self.render_pass = unsafe {
            self.device
                .as_ref()
                .unwrap()
                .create_render_pass(&render_pass_info, None)
                .expect("create_render_pass error")
        };
    }

    ///
    /// 创建管线
    ///
    /// 可编程管线阶段
    ///
    pub(crate) fn create_graphics_pipeline(&mut self) {
        let vert_shader_code = Self::read_file(Path::new("src/shader/vert.spv"));
        let frag_shader_code = Self::read_file(Path::new("src/shader/frag.spv"));

        let vert_shader_module = self.create_shader_module(&vert_shader_code);
        let frag_shader_module = self.create_shader_module(&frag_shader_code);
        let name = CString::new("main").unwrap();
        // 顶点
        let vert_shader_stage_info = PipelineShaderStageCreateInfo::builder()
            .stage(ShaderStageFlags::VERTEX)
            .module(vert_shader_module)
            .name(name.as_c_str())
            .build();

        // 片元
        let frag_shader_stage_info = PipelineShaderStageCreateInfo::builder()
            .stage(ShaderStageFlags::FRAGMENT)
            .module(frag_shader_module)
            .name(name.as_c_str())
            .build();

        let shader_stages = vec![vert_shader_stage_info, frag_shader_stage_info];

        //顶点输入
        let vertex_input_info = PipelineVertexInputStateCreateInfo::default();
        //输入组件
        let input_assembly = PipelineInputAssemblyStateCreateInfo::builder()
            // VK_PRIMITIVE_TOPOLOGY_POINT_LIST 顶点
            // VK_PRIMITIVE_TOPOLOGY_LINE_LIST 每2个点组成一条线，而不会重复
            // VK_PRIMITIVE_TOPOLOGY_LINE_STRIP 每行的终点用作下一行的起点
            // VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST 每3个顶点组成三角形，而无需重用
            // VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP 每个三角形的第二个和第三个顶点用作下一个三角形的前两个顶点
            .topology(PrimitiveTopology::TRIANGLE_LIST)
            //. If you set the primitiveRestartEnable member to VK_TRUE, then it's possible to break up lines and triangles in the _STRIP topology modes by using a special index of 0xFFFF or 0xFFFFFFFF.
            .primitive_restart_enable(false)
            .build();

        //视口定义了从图像到帧缓冲区的转换，而裁剪矩形定义了实际存储像素的区域
        //裁剪矩形之外的所有像素将被光栅化器丢弃。

        //视口
        let viewport = Viewport::builder()
            .x(0f32)
            .y(0f32)
            .width(self.swap_chain_extent.width as f32)
            .height(self.swap_chain_extent.height as f32)
            //minDepth和maxDepth值指定深度值的范围来使用的帧缓冲。这些值必须在[0.0f, 1.0f]范围内
            //minDepth可以大于maxDepth
            //如果您没有做任何特别的事情，则应遵循0.0fand 的标准值1.0f。
            .min_depth(0f32)
            .max_depth(1f32)
            .build();

        //请记住，交换链及其图像的大小可能与窗口的WIDTH和HEIGHT有所不同。交换链图像稍后将用作帧缓冲区，因此我们应保持其大小一致。
        //裁剪
        let scissor = Rect2D::builder()
            .offset(Offset2D { x: 0, y: 0 })
            .extent(self.swap_chain_extent)
            .build();

        let viewport_state = PipelineViewportStateCreateInfo::builder()
            //在某些图形卡上可以使用多个视口和剪刀矩形
            //使用多个功能需要启用GPU功能（请参阅逻辑设备创建）。
            .viewports(&[viewport])
            .scissors(&[scissor])
            .build();

        //光栅化器
        let rasterizer = PipelineRasterizationStateCreateInfo::builder()
            .depth_clamp_enable(false)
            .rasterizer_discard_enable(false)
            // VK_POLYGON_MODE_FILL 用片段填充多边形区域
            // VK_POLYGON_MODE_LINE 多边形的边缘绘制为线
            // VK_POLYGON_MODE_POINT 将多边形顶点绘制为点
            .polygon_mode(PolygonMode::FILL)
            //根据片段的数量描述线条的粗细
            .line_width(1.0f32)
            //确定要使用的面部剔除类型。您可以禁用剔除，删除正面，删除背面或同时禁用这两者
            .cull_mode(CullModeFlags::BACK)
            //指定将面视为正面的顶点顺序，可以是顺时针或逆时针。
            .front_face(FrontFace::CLOCKWISE)
            //栅格化器可以通过添加恒定值或基于片段的斜率对深度值进行偏置来更改深度值。有时用于阴影贴图，但我们不会使用它。只需设置depthBiasEnable为即可VK_FALSE。
            .depth_bias_enable(false)
            .depth_bias_constant_factor(0.0f32)
            .depth_bias_clamp(0.0f32)
            .depth_bias_slope_factor(0.0f32)
            .build();

        //多重采样
        //我们将在后面的章节中重新介绍多重采样，现在让我们禁用它
        let multisampling = PipelineMultisampleStateCreateInfo::builder()
            .sample_shading_enable(false)
            .rasterization_samples(SampleCountFlags::TYPE_1)
            .min_sample_shading(1.0f32)
            .alpha_to_coverage_enable(false)
            .alpha_to_one_enable(false)
            .build();

        //深度和模板测试
        //我们将在深度缓冲一章中再次介绍它

        //色彩融合
        //片段着色器返回颜色后，需要将其与帧缓冲区中已经存在的颜色合并。这种转换称为颜色混合，有两种方法可以实现：
        //混合新旧值以产生最终颜色
        //使用按位运算将新旧值合并
        let color_blend_attachment = PipelineColorBlendAttachmentState::builder()
            .color_write_mask(
                ColorComponentFlags::R
                    | ColorComponentFlags::G
                    | ColorComponentFlags::B
                    | ColorComponentFlags::A,
            )
            .blend_enable(false)
            .src_color_blend_factor(BlendFactor::ONE)
            .dst_color_blend_factor(BlendFactor::ZERO)
            .color_blend_op(BlendOp::ADD)
            .src_alpha_blend_factor(BlendFactor::ONE)
            .dst_alpha_blend_factor(BlendFactor::ZERO)
            .alpha_blend_op(BlendOp::ADD)
            .build();

        // 这种每帧缓冲区结构允许您配置颜色混合的第一种方法。使用以下伪代码可以最好地演示将要执行的操作：
        // if (blendEnable) {
        //     finalColor.rgb = (srcColorBlendFactor * newColor.rgb) <colorBlendOp> (dstColorBlendFactor * oldColor.rgb);
        //     finalColor.a = (srcAlphaBlendFactor * newColor.a) <alphaBlendOp> (dstAlphaBlendFactor * oldColor.a);
        // } else {
        //     finalColor = newColor;
        // }

        // finalColor = finalColor & colorWriteMask;

        // 如果blendEnable设置为VK_FALSE，则来自片段着色器的新颜色将未经修改地传递。否则，将执行两次混合操作以计算新的颜色。产生的颜色与colorWriteMask进行“与”运算， 以确定实际通过哪些通道。
        // 使用颜色混合的最常见方法是实现Alpha混合，在此我们希望根据新颜色的不透明度将新颜色与旧颜色混合。该 finalColor则应计算如下：

        // finalColor.rgb = newAlpha * newColor + (1 - newAlpha) * oldColor;
        // finalColor.a = newAlpha.a;

        // 这可以通过以下参数来完成：
        // colorBlendAttachment.blendEnable = VK_TRUE;
        // colorBlendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
        // colorBlendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
        // colorBlendAttachment.colorBlendOp = VK_BLEND_OP_ADD;
        // colorBlendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
        // colorBlendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
        // colorBlendAttachment.alphaBlendOp = VK_BLEND_OP_ADD;

        let color_blending = PipelineColorBlendStateCreateInfo::builder()
            .logic_op_enable(false)
            .logic_op(LogicOp::COPY)
            .attachments(&[color_blend_attachment])
            .blend_constants([0.0f32, 0.0f32, 0.0f32, 0.0f32])
            .build();

        //动态状态

        //管道布局
        let pipeline_layout_info = PipelineLayoutCreateInfo::builder().build();

        unsafe {
            self.pipeline_layout = self
                .device
                .as_ref()
                .unwrap()
                .create_pipeline_layout(&pipeline_layout_info, None)
                .expect("create_pipeline_layout")
        };

        let pipeline_info = GraphicsPipelineCreateInfo::builder()
            .stages(&shader_stages)
            .vertex_input_state(&vertex_input_info)
            .input_assembly_state(&input_assembly)
            .viewport_state(&viewport_state)
            .rasterization_state(&rasterizer)
            .multisample_state(&multisampling)
            .color_blend_state(&color_blending)
            .layout(self.pipeline_layout)
            .render_pass(self.render_pass)
            .build();

        unsafe {
            self.graphics_pipeline = self
                .device
                .as_ref()
                .unwrap()
                .create_graphics_pipelines(PipelineCache::null(), &[pipeline_info], None)
                .expect("create_graphics_pipelines error")[0]
        };

        unsafe {
            self.device
                .as_ref()
                .unwrap()
                .destroy_shader_module(frag_shader_module, None);

            self.device
                .as_ref()
                .unwrap()
                .destroy_shader_module(vert_shader_module, None);
        };
    }

    ///
    /// 创建帧缓冲
    ///
    pub(crate) fn create_framebuffers(&mut self) {
        for (_i, &swap_chain_image_view) in self.swap_chain_image_views.iter().enumerate() {
            let attachments = vec![ImageView::from(swap_chain_image_view)];

            let framebuffer_info = FramebufferCreateInfo::builder()
                .render_pass(self.render_pass)
                .attachments(&attachments)
                .width(self.swap_chain_extent.width)
                .height(self.swap_chain_extent.height)
                .layers(1)
                .build();

            unsafe {
                self.swap_chain_framebuffers.push(
                    self.device
                        .as_ref()
                        .unwrap()
                        .create_framebuffer(&framebuffer_info, None)
                        .expect("create_framebuffer error"),
                );
            };
        }
    }

    pub(crate) fn create_command_pool(&mut self) {
        let physical_device = self.physical_device.as_ref().unwrap();
        let indices = self.find_queue_families(physical_device);

        //COMMAND_POOL_CREATE_TRANSIENT_BIT：提示命令缓冲区经常用新命令重新记录（可能会更改内存分配行为）
        //COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT：允许单独重新记录命令缓冲区，没有此标志，则必须将它们全部一起重置
        let pool_info = CommandPoolCreateInfo::builder()
            .queue_family_index(indices.graphics_family.unwrap())
            .build();

        self.command_pool = unsafe {
            self.device
                .as_ref()
                .unwrap()
                .create_command_pool(&pool_info, None)
                .expect("command_pool error")
        };
    }

    pub(crate) fn create_command_buffers(&mut self) {
        let alloc_info = CommandBufferAllocateInfo::builder()
            .command_pool(self.command_pool)
            .level(CommandBufferLevel::PRIMARY)
            .command_buffer_count(self.swap_chain_framebuffers.len() as u32)
            .build();

        self.command_buffers = unsafe {
            self.device
                .as_ref()
                .unwrap()
                .allocate_command_buffers(&alloc_info)
                .expect("allocate_command_buffers error")
        };
        unsafe {
            for (i, &command_buffer) in self.command_buffers.iter().enumerate() {
                let begin_info = CommandBufferBeginInfo::builder().build();

                self.device
                    .as_ref()
                    .unwrap()
                    .begin_command_buffer(command_buffer, &begin_info)
                    .expect("begin_command_buffer error");

                let clear_values = [ClearValue {
                    color: ClearColorValue {
                        float32: [0.0, 0.0, 0.0, 1.0],
                    },
                }];

                let render_pass_info = RenderPassBeginInfo::builder()
                    .render_pass(self.render_pass)
                    .framebuffer(self.swap_chain_framebuffers[i])
                    .render_area(Rect2D {
                        offset: Offset2D { x: 0, y: 0 },
                        extent: self.swap_chain_extent,
                    })
                    .clear_values(&clear_values)
                    .build();

                self.device.as_ref().unwrap().cmd_begin_render_pass(
                    command_buffer,
                    &render_pass_info,
                    SubpassContents::INLINE,
                );

                self.device.as_ref().unwrap().cmd_bind_pipeline(
                    command_buffer,
                    PipelineBindPoint::GRAPHICS,
                    self.graphics_pipeline,
                );

                self.device
                    .as_ref()
                    .unwrap()
                    .cmd_draw(command_buffer, 3, 1, 0, 0);

                self.device
                    .as_ref()
                    .unwrap()
                    .cmd_end_render_pass(command_buffer);

                self.device
                    .as_ref()
                    .unwrap()
                    .end_command_buffer(command_buffer)
                    .expect("end_command_buffer error");
            }
        }
    }

    ///
    /// 创建着色器模块
    ///
    pub(crate) fn create_shader_module(&self, code: &Vec<u32>) -> ShaderModule {
        let shader = ShaderModuleCreateInfo::builder().code(code).build();
        unsafe {
            self.device
                .as_ref()
                .unwrap()
                .create_shader_module(&shader, None)
                .expect("create_shader_module error")
        }
    }

    ///
    /// 校验交换链是否支持
    ///
    pub(crate) fn query_swap_chain_support(
        &self,
        device: &PhysicalDevice,
    ) -> SwapChainSupportDetails {
        let surface_capabilities = unsafe {
            self.surface_loader
                .as_ref()
                .unwrap()
                .get_physical_device_surface_capabilities(*device, self.surface.unwrap())
                .expect("get_physical_device_surface_capabilities error")
        };

        let formats = unsafe {
            self.surface_loader
                .as_ref()
                .unwrap()
                .get_physical_device_surface_formats(*device, self.surface.unwrap())
                .expect("get_physical_device_surface_formats error")
        };

        let present_modes = unsafe {
            self.surface_loader
                .as_ref()
                .unwrap()
                .get_physical_device_surface_present_modes(*device, self.surface.unwrap())
                .expect("get_physical_device_surface_present_modes error")
        };

        let details = SwapChainSupportDetails {
            capabilities: surface_capabilities,
            formats,
            present_modes,
        };

        details
    }

    pub(crate) fn is_device_suitable(&mut self, device: &PhysicalDevice) -> bool {
        let indices = self.find_queue_families(device);

        let extensions_supported = self.check_device_extension_support(device);

        let mut swap_chain_adequate = false;
        if extensions_supported {
            let swap_chain_support = self.query_swap_chain_support(device);
            swap_chain_adequate = !swap_chain_support.formats.is_empty()
                && !swap_chain_support.present_modes.is_empty();
        }

        return indices.is_complete() && extensions_supported && swap_chain_adequate;
    }

    ///
    /// format成员指定颜色通道和类型
    /// SRGB_NONLINEAR_KHR标志指示是否支持SRGB颜色空间
    ///
    /// 对于色彩空间，我们将使用SRGB（如果可用）
    /// @see https://stackoverflow.com/questions/12524623/what-are-the-practical-differences-when-working-with-colors-in-a-linear-vs-a-no
    ///
    pub(crate) fn choose_swap_surface_format(
        &self,
        available_formats: Vec<SurfaceFormatKHR>,
    ) -> SurfaceFormatKHR {
        for (_i, format) in available_formats.iter().enumerate() {
            if format.format == Format::B8G8R8A8_UNORM
                && format.color_space == ColorSpaceKHR::SRGB_NONLINEAR
            {
                return *format;
            }
        }

        //那么我们可以根据它们的"好"的程度开始对可用格式进行排名，但是在大多数情况下，只需要使用指定的第一种格式就可以了。
        return available_formats[0];
    }

    ///
    /// 垂直空白间隙 vertical blank interval(VBI)，类比显示器显示一张画面的方式:由画面的左上角开始以交错的方式最后扫描至画面的右下角.，这样就完成了一张画面的显示,，然后电子束移回去左上角， 以进行下一张画面的显示。
    ///
    /// 显示模式可以说是交换链最重要的设置，因为它代表了在屏幕上显示图像的实际条件
    /// Vulkan有四种可能的模式：
    ///
    /// VK_PRESENT_MODE_IMMEDIATE_KHR：您的应用程序提交的图像会立即传输到屏幕上，这可能会导致撕裂。
    /// VK_PRESENT_MODE_FIFO_KHR：交换链是一个队列，当刷新显示时，显示器从队列的前面获取图像，并且程序将渲染的图像插入队列的后面。如果队列已满，则程序必须等待。这与现代游戏中的垂直同步最为相似。刷新显示的那一刻被称为“垂直空白间隙”。
    /// VK_PRESENT_MODE_FIFO_RELAXED_KHR：仅当应用程序延迟并且队列在最后一个垂直空白间隙处为空时，此模式才与前一个模式不同。当图像最终到达时，将立即传输图像，而不是等待下一个垂直空白间隙。这可能会导致可见的撕裂。
    /// VK_PRESENT_MODE_MAILBOX_KHR：这是第二种模式的另一种形式。当队列已满时，不会阻塞应用程序，而是将已经排队的图像替换为更新的图像。此模式可用于实现三重缓冲，与使用双缓冲的标准垂直同步相比，它可以避免撕裂，并显着减少了延迟问题。
    ///
    ///
    pub(crate) fn choose_swap_present_mode(
        &self,
        available_present_modes: Vec<PresentModeKHR>,
    ) -> PresentModeKHR {
        for (_i, present_mode) in available_present_modes.iter().enumerate() {
            if present_mode.as_raw() == PresentModeKHR::MAILBOX.as_raw() {
                return *present_mode;
            }
        }

        return PresentModeKHR::FIFO;
    }

    ///
    /// 交换范围是交换链图像的分辨率，它几乎始终等于我们要绘制到的窗口的分辨率
    ///
    pub(crate) fn choose_swap_extent(&self, capabilities: &SurfaceCapabilitiesKHR) -> Extent2D {
        if capabilities.current_extent.width != u32::max_value() {
            return capabilities.current_extent;
        } else {
            use std::cmp::{max, min};

            let mut actual_extent = Extent2D::builder().width(WIDTH).height(HEIGHT).build();
            actual_extent.width = max(
                capabilities.min_image_extent.width,
                min(capabilities.min_image_extent.width, actual_extent.width),
            );

            actual_extent.height = max(
                capabilities.min_image_extent.height,
                min(capabilities.min_image_extent.height, actual_extent.height),
            );

            return actual_extent;
        };
    }

    ///
    /// 校验扩展支持情况
    ///
    pub(crate) fn check_device_extension_support(&self, device: &PhysicalDevice) -> bool {
        let device_extension_properties: Vec<ExtensionProperties> = unsafe {
            self.instance
                .as_ref()
                .unwrap()
                .enumerate_device_extension_properties(*device)
                .expect("failed to get device extension properties.")
        };

        let mut extensions = DEVICE_EXTENSIONES.clone().to_vec();
        for (_i, dep) in device_extension_properties.iter().enumerate() {
            // nightly
            // todo https://doc.rust-lang.org/std/vec/struct.Vec.html#method.remove_item

            let index = extensions
                .iter()
                .position(|x| *x == Self::char2str(&dep.extension_name));

            if let Some(index) = index {
                extensions.remove(index);
            }
        }

        extensions.is_empty()
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
                self.device
                    .as_ref()
                    .unwrap()
                    .destroy_command_pool(self.command_pool, None);

                for (_i, &framebuffer) in self.swap_chain_framebuffers.iter().enumerate() {
                    self.device
                        .as_ref()
                        .unwrap()
                        .destroy_framebuffer(framebuffer, None);
                }

                self.device
                    .as_ref()
                    .unwrap()
                    .destroy_pipeline(self.graphics_pipeline, None);

                self.device
                    .as_ref()
                    .unwrap()
                    .destroy_pipeline_layout(self.pipeline_layout, None);

                self.device
                    .as_ref()
                    .unwrap()
                    .destroy_render_pass(self.render_pass, None);

                for &image_view in self.swap_chain_image_views.iter() {
                    self.device
                        .as_ref()
                        .unwrap()
                        .destroy_image_view(image_view, None);
                }

                self.swap_chain_loader
                    .as_ref()
                    .unwrap()
                    .destroy_swapchain(self.swap_chain, None);

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

    pub(crate) fn read_file(filename: &Path) -> Vec<u32> {
        use std::{fs::File, io::Read};

        let spv_file =
            File::open(filename).expect(&format!("Failed to find spv file at {:?}", filename));
        let bytes_code: Vec<u8> = spv_file.bytes().filter_map(|byte| byte.ok()).collect();

        ash::util::read_spv(&mut Cursor::new(bytes_code)).expect("read_spv error")
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
/// 交换链的一般目的是使图像的显示与屏幕的刷新率同步。
/// but the general purpose of the swap chain is to synchronize the presentation of images with the refresh rate of the screen.
///
/// 现在，我们可以结合前几章中的所有结构和对象来创建图形管道！快速回顾一下，这是我们现在拥有的对象的类型：
/// 着色器阶段：定义图形管线可编程阶段功能的着色器模块
/// 固定功能状态：定义管道固定功能阶段的所有结构，例如输入组件，光栅化器，视口和颜色混合
/// 管线布局：着色器引用的统一值和推动值，可以在绘制时进行更新
/// 渲染阶段：管道阶段引用的附件及其用法
///
///
pub fn main() {
    let events = EventLoop::new();
    let mut hello = HelloTriangleApplication::default();
    let win = hello.run(&events);
    hello.main_loop(events, win);
}

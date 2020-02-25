//!
//!
//! @see https://vulkan-tutorial.com/Vertex_buffers/Vertex_buffer_creation
//! @see https://www.khronos.org/registry/vulkan/specs/1.1-extensions/html/vkspec.html#VK_EXT_debug_utils
//! cargo run --features=debug loading_models
//!
//! 注：本教程所有的英文注释都是有google翻译而来。如有错漏,请告知我修改
//!
//! Note: All English notes in this tutorial are translated from Google. If there are errors and omissions, please let me know
//!
//! The MIT License (MIT)
//!
#[cfg(target_os = "windows")]
use ash::extensions::khr::Win32Surface;
use ash::{
    extensions::{
        ext::DebugUtils,
        khr::{Surface, Swapchain},
    },
    version::{DeviceV1_0, EntryV1_0, InstanceV1_0},
    vk::*,
    Entry, Instance,
};
use image::GenericImageView;
use nal::{Matrix, Matrix4, Perspective3, Point3, Vector2, Vector3};
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

///
/// 统一缓冲区对象UBO
///
#[repr(C, align(16))]
struct UniformBufferObject {
    //pub foo: Vector2<f32>,
    ///
    /// 模型
    ///
    pub model: Matrix4<f32>,

    ///
    /// 视图
    ///
    pub view: Matrix4<f32>,

    ///
    /// 投影
    ///
    pub proj: Perspective3<f32>,
}

impl Drop for UniformBufferObject {
    fn drop(&mut self) {
        info!("drop ubo");
    }
}

#[repr(C)]
struct Vertex {
    pub pos: Vector3<f32>,
    pub color: Vector3<f32>,
    pub text_coord: Vector2<f32>,
}

impl Vertex {
    //告诉Vulkan一旦将数据格式上传到GPU内存后如何将其传递给顶点着色器
    pub fn get_binding_description() -> VertexInputBindingDescription {
        //顶点绑定描述了整个顶点从内存中加载数据的速率。它指定数据条目之间的字节数，以及是否在每个顶点之后或在每个实例之后移至下一个数据条目。
        let mut binding_description = VertexInputBindingDescription::default();
        binding_description.binding = 0;
        binding_description.stride = std::mem::size_of::<Vertex>() as u32;
        //我们所有的每个顶点数据都包装在一个数组中，因此我们只需要一个绑定。该binding参数指定绑定数组中绑定的索引。该stride参数指定从一个条目到下一个条目的字节数，并且该inputRate参数可以具有以下值之一：
        //VERTEX_INPUT_RATE_VERTEX：每个顶点后移至下一个数据条目
        //VERTEX_INPUT_RATE_INSTANCE：每个实例后移至下一个数据条目
        binding_description.input_rate = VertexInputRate::VERTEX;

        binding_description
    }

    pub fn get_attribute_descriptions() -> [VertexInputAttributeDescription; 3] {
        let mut attribute_descriptions = [
            VertexInputAttributeDescription::default(),
            VertexInputAttributeDescription::default(),
            VertexInputAttributeDescription::default(),
        ];

        //该binding参数告诉Vulkan每个顶点数据来自哪个绑定。该location参数引用location顶点着色器中输入的指令
        //带有位置的顶点着色器中的输入0是位置，该位置具有两个32位浮点分量。
        attribute_descriptions[0].binding = 0;
        attribute_descriptions[0].location = 0;
        //该format参数描述该属性的数据类型
        //float： VK_FORMAT_R32_SFLOAT
        //vec2： VK_FORMAT_R32G32_SFLOAT
        //vec3： VK_FORMAT_R32G32B32_SFLOAT
        //vec4： VK_FORMAT_R32G32B32A32_SFLOAT
        //您应该使用颜色通道数量与着色器数据类型中的组件数量匹配的格式
        //如果通道数少于组件数，则BGA组件将使用默认值(0, 0, 1)。颜色类型（SFLOAT，UINT，SINT）和比特宽度也应与着色器输入的类型。
        //ivec2：VK_FORMAT_R32G32_SINT，由32位有符号整数组成的2分量向量
        //uvec4：VK_FORMAT_R32G32B32A32_UINT，一个由32位无符号整数组成的4分量向量
        //double：VK_FORMAT_R64_SFLOAT，双精度（64位）浮点数
        attribute_descriptions[0].format = Format::R32G32B32_SFLOAT;
        attribute_descriptions[0].offset = 0;

        attribute_descriptions[1].binding = 0;
        attribute_descriptions[1].location = 1;
        attribute_descriptions[1].format = Format::R32G32B32_SFLOAT;
        attribute_descriptions[1].offset = std::mem::size_of::<Vector3<f32>>() as u32;

        attribute_descriptions[2].binding = 0;
        attribute_descriptions[2].location = 2;
        attribute_descriptions[2].format = Format::R32G32_SFLOAT;
        attribute_descriptions[2].offset =
            std::mem::size_of::<Vector3<f32>>() as u32 + std::mem::size_of::<Vector3<f32>>() as u32;

        attribute_descriptions
    }
}

///
/// rust不允许随意使用一个静态的结构体数据
/// 我们通过方法获取这个数据
///
fn generator_vertices() {
    unsafe {
        START.call_once(|| {
            //我们将修改顶点数据并添加索引数据以绘制一个矩形
            VERTICES = std::mem::transmute(Box::new(vec![
                Vertex {
                    pos: Vector3::new(-0.5f32, -0.5f32, 0.0f32),
                    color: Vector3::new(1.0, 0.0, 0.0),
                    text_coord: Vector2::new(1.0, 0.0),
                },
                Vertex {
                    pos: Vector3::new(0.5f32, -0.5f32, 0.0),
                    color: Vector3::new(0.0, 1.0, 0.0),
                    text_coord: Vector2::new(0.0, 0.0),
                },
                Vertex {
                    pos: Vector3::new(0.5f32, 0.5f32, 0.0),
                    color: Vector3::new(0.0, 0.0, 1.0),
                    text_coord: Vector2::new(0.0, 1.0),
                },
                Vertex {
                    pos: Vector3::new(-0.5f32, 0.5f32, 0.0),
                    color: Vector3::new(1.0, 1.0, 1.0),
                    text_coord: Vector2::new(1.0, 1.0),
                },
                //============================================
                Vertex {
                    pos: Vector3::new(-0.5f32, -0.5f32, -0.5f32),
                    color: Vector3::new(1.0, 0.0, 0.0),
                    text_coord: Vector2::new(1.0, 0.0),
                },
                Vertex {
                    pos: Vector3::new(0.5f32, -0.5f32, -0.5),
                    color: Vector3::new(0.0, 1.0, 0.0),
                    text_coord: Vector2::new(0.0, 0.0),
                },
                Vertex {
                    pos: Vector3::new(0.5f32, 0.5f32, -0.5),
                    color: Vector3::new(0.0, 0.0, 1.0),
                    text_coord: Vector2::new(0.0, 1.0),
                },
                Vertex {
                    pos: Vector3::new(-0.5f32, 0.5f32, -0.5),
                    color: Vector3::new(1.0, 1.0, 1.0),
                    text_coord: Vector2::new(1.0, 1.0),
                },
            ]));
        });
    }
}

///
/// 索引
///
/// 可以使用uint16_t或uint32_t作为索引缓冲区，具体取决于中的条目数vertices。我们uint16_t现在可以坚持使用，因为我们使用的少于65535个唯一顶点。
///
const INDICES: [u32; 12] = [0, 1, 2, 2, 3, 0, 4, 5, 6, 6, 7, 4];

static mut VERTICES: *mut Vec<Vertex> = std::ptr::null_mut::<Vec<Vertex>>();
static START: std::sync::Once = std::sync::Once::new();

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
///
/// 该常量定义应同时处理多少帧
///
const MAX_FRAMES_IN_FLIGHT: usize = 2;

///
/// 模型路径
///
const MODEL_PATH: &'static str = "src/models/chalet.obj";

///
/// 模型纹理路径
///
const TEXTURE_PATH: &'static str = "src/textures/chalet.jpg";

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
    /// 所有描述符绑定都组合到一个DescriptorSetLayout对象中
    ///
    pub(crate) descriptor_set_layout: DescriptorSetLayout,

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
    /// 命令缓冲
    ///
    pub(crate) command_buffers: Vec<CommandBuffer>,

    ///
    /// 我们将需要一个信号量来表示已获取图像并可以进行渲染
    /// 每帧应具有自己的一组信号量
    ///
    pub(crate) image_available_semaphores: Vec<Semaphore>,

    ///
    /// 另一个信号量则表示已完成渲染并可以进行呈现
    /// 每帧应具有自己的一组信号量
    ///
    pub(crate) render_finished_semaphores: Vec<Semaphore>,

    ///
    ///
    ///
    pub(crate) inflight_fences: Vec<Fence>,

    ///
    /// 我们将立即有一个同步对象等待新帧可以使用该图像。
    ///
    pub(crate) images_inflight: Vec<Fence>,

    ///
    /// 要每次都使用正确的一组信号量，我们需要跟踪当前帧。我们将为此使用帧索引
    ///
    pub(crate) current_frame: i32,

    ///
    /// 顶点数组
    ///
    pub(crate) vertices: Vec<Vertex>,

    ///
    /// 索引数组
    ///
    pub(crate) indices: Vec<u32>,

    ///
    /// 顶点缓冲区句柄
    ///
    pub(crate) vertex_buffer: Buffer,

    ///
    /// 创建一个类成员以将句柄存储到内存中并使用进行分配
    ///
    pub(crate) vertex_buffer_memory: DeviceMemory,

    ///
    /// 索引缓冲区句柄
    ///
    pub(crate) index_buffer: Buffer,

    ///
    /// 创建一个类成员以将句柄存储到内存中并使用进行分配
    ///
    pub(crate) index_buffer_memory: DeviceMemory,

    ///
    /// UBO
    ///
    pub(crate) uniform_buffers: Vec<Buffer>,

    ///
    /// 创建一个类成员以将句柄存储到内存中并使用进行分配
    ///
    pub(crate) uniform_buffers_memory: Vec<DeviceMemory>,

    ///
    /// 描述符池
    ///
    pub(crate) descriptor_pool: DescriptorPool,

    ///
    /// 描述符集
    ///
    pub(crate) descriptor_sets: Vec<DescriptorSet>,

    ///
    /// 纹理图像
    ///
    pub(crate) texture_image: Image,

    ///
    /// 纹理图像内存
    ///
    pub(crate) textre_image_memory: DeviceMemory,

    ///
    /// 纹理视图
    ///
    pub(crate) texture_image_view: ImageView,

    ///
    /// 采样器
    ///
    pub(crate) texture_sampler: Sampler,

    ///
    /// 深度纹理图像
    ///
    pub(crate) depth_image: Image,

    ///
    /// 深度纹理图像内存
    ///
    pub(crate) depth_image_memory: DeviceMemory,

    ///
    /// 深度纹理视图
    ///
    pub(crate) depth_image_view: ImageView,
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
        self.create_descriptor_set_layout();
        self.create_graphics_pipeline();
        self.create_command_pool();
        self.create_depth_resources();
        self.create_framebuffers();
        self.create_texture_image();
        self.create_texture_image_view();
        self.create_texture_sampler();
        self.load_model();
        self.create_vertex_buffer();
        self.create_index_buffer();
        self.create_uniform_buffers();
        self.create_descriptor_pool();
        self.create_descriptor_sets();
        self.create_command_buffers();
        self.create_sync_objects();
    }

    ///
    /// 为依赖交换链或窗口大小的对象创建一个新的recreateSwapChain调用函数createSwapChain以及所有创建函数。
    ///
    pub(crate) fn recreate_swap_chain(&mut self) {
        //我们之所以称之为vkDeviceWaitIdle，是因为与上一章一样，我们不应该接触可能仍在使用的资源
        unsafe {
            self.device
                .as_ref()
                .unwrap()
                .device_wait_idle()
                .expect("device_wait_idle error");
        }

        //显然，我们要做的第一件事是重新创建交换链本身。需要重新创建图像视图，因为它们直接基于交换链图像。需要重新创建渲染通道，因为它取决于交换链图像的格式。交换链图像格式在诸如窗口调整大小之类的操作期间很少发生更改，但仍应处理。在图形管道创建期间指定了视口和剪刀矩形的大小，因此也需要重建管道。通过为视口和剪刀矩形使用动态状态，可以避免这种情况。最后，帧缓冲区和命令缓冲区也直接取决于交换链映像。
    }

    ///
    /// 为了确保在重新创建它们之前清除这些对象的旧版本，我们应该将一些清除代码移到一个单独的函数中，可以从该recreateSwapChain函数调用该函数。让我们称之为 cleanupSwapChain
    ///
    /// 我们将将交换链刷新中重新创建的所有对象的清除代码从cleanup移至cleanupSwapChain：
    ///
    pub(crate) fn cleanup_swap_chain(&mut self) {
        unsafe {
            let device = self.device.as_ref().unwrap();

            for i in 0..self.swap_chain_images.len() {
                device.destroy_buffer(self.uniform_buffers[i], None);

                device.free_memory(self.uniform_buffers_memory[i], None);
            }

            device.destroy_descriptor_pool(self.descriptor_pool, None);

            for (_i, &framebuffer) in self.swap_chain_framebuffers.iter().enumerate() {
                device.destroy_framebuffer(framebuffer, None);
            }

            //我们可以从头开始重新创建命令池，但这很浪费。相反，我选择使用该vkFreeCommandBuffers函数清理现有的命令缓冲区 。这样
            //我们可以重用现有池来分配新的命令缓冲区。
            device.free_command_buffers(self.command_pool, &self.command_buffers);

            device.destroy_pipeline(self.graphics_pipeline, None);

            device.destroy_pipeline_layout(self.pipeline_layout, None);

            device.destroy_render_pass(self.render_pass, None);

            for &image_view in self.swap_chain_image_views.iter() {
                device.destroy_image_view(image_view, None);
            }

            self.swap_chain_loader
                .as_ref()
                .unwrap()
                .destroy_swapchain(self.swap_chain, None);
        }
    }

    ///
    /// 主循环
    /// Main loop
    ///
    /// 为了在出现错误或窗口关闭之前保持应用程序运行，我们需要向函数添加事件循环
    /// In order to keep the application running until an error occurs or the window closes, we need to add an event loop to the function
    ///
    pub(crate) fn main_loop(mut self, event_loop: EventLoop<()>, window: Window) {
        event_loop.run(move |event, _, control_flow| {
            *control_flow = ControlFlow::Wait;

            match event {
                Event::LoopDestroyed => {
                    unsafe {
                        self.device
                            .as_ref()
                            .unwrap()
                            .device_wait_idle()
                            .expect("Failed to wait device idle!")
                    };
                }
                Event::WindowEvent { event, .. } => match event {
                    WindowEvent::CloseRequested => {
                        *control_flow = ControlFlow::Exit;
                    }
                    _ => (),
                },
                Event::MainEventsCleared => {
                    window.request_redraw();
                }
                Event::RedrawRequested(_) => {
                    self.draw_frame();
                }
                _ => (),
            }
        });
    }

    ///
    /// 该drawFrame函数将执行以下操作：
    /// 从交换链获取图像
    /// 以该图像作为附件在帧缓冲区中执行命令缓冲区
    /// 将图像返回交换链进行演示
    /// 这些事件中的每一个都使用单个函数调用设置为运动中的，但是它们是异步执行的。函数调用将在操作实际完成之前返回，并且执行顺序也未定义。不幸的是，因为每个操作都取决于上一个操作。
    ///
    /// 有两种同步交换链事件的方式：阑珊和信号量
    ///
    /// 我们已经实现了所有必需的同步，以确保排队的工作帧不超过两个，并且这些帧不会意外地使用相同的图像。请注意，对于代码的其他部分（如最终清理），可以依赖更粗糙的同步，例如vkDeviceWaitIdle。您应该根据性能要求决定使用哪种方法。
    ///
    /// 同步相关，请思看：
    /// https://github.com/KhronosGroup/Vulkan-Docs/wiki/Synchronization-Examples#swapchain-image-acquire-and-present
    ///
    fn draw_frame(&mut self) {
        unsafe {
            //该vkWaitForFences函数接收一组栅栏，并在返回之前等待其中的任何一个或全部都发信号。
            self.device
                .as_ref()
                .unwrap()
                .wait_for_fences(
                    &[self.inflight_fences[self.current_frame as usize]],
                    //在VTRUE我们路过这里表示我们要等待所有栅栏，但在一个单一的情况下，它显然并不重要。
                    true,
                    u64::max_value(),
                )
                .expect("wait_for_fences error");

            let acquire_next_image = self.swap_chain_loader.as_ref().unwrap().acquire_next_image(
                self.swap_chain,
                u64::max_value(),
                self.image_available_semaphores[self.current_frame as usize],
                Fence::null(),
            );

            if let Err(err) = acquire_next_image {
                //ERROR_OUT_OF_DATE_KHR：交换链变得与曲面不兼容，不能再用于渲染。通常在调整窗口大小之后发生。
                if err == Result::ERROR_OUT_OF_DATE_KHR {
                    //如果尝试获取图像时交换链已过期，则无法再显示该图像。因此，我们应该立即重新创建交换链，并在下一个drawFrame调用中重试。
                    self.recreate_swap_chain();
                    return;
                }

                panic!("error is:{:?}", err);
            }

            let acquire_next_image = acquire_next_image.expect("error acquire_next_image 1");
            let image_index = acquire_next_image.0;

            if acquire_next_image.1 {
                //SUBOPTIMAL_KHR：交换链仍可用于成功显示在表面上，但是表面特性不再完全匹配。
                info!("SUBOPTIMAL_KHR：交换链仍可用于成功显示在表面上，但是表面特性不再完全匹配。");
            }

            self.update_uniform_buffer(image_index);

            //我们将修改drawFrame以等待正在使用刚刚为新帧分配的图像的任何先前帧
            //检查前一帧是否正在使用此图像（即，有其栏珊等待）
            if self.images_inflight[image_index as usize] != Fence::null() {
                self.device
                    .as_ref()
                    .unwrap()
                    .wait_for_fences(
                        &[self.images_inflight[image_index as usize]],
                        true,
                        u64::max_value(),
                    )
                    .expect("wait_for_fences2 error");
            }

            //将图像标记为此帧正在使用
            self.images_inflight[image_index as usize] =
                self.inflight_fences[self.current_frame as usize];

            let wait_semaphores = [self.image_available_semaphores[self.current_frame as usize]];
            let single_semaphores = [self.render_finished_semaphores[self.current_frame as usize]];
            let wait_stages = [PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
            let submit_info = SubmitInfo::builder()
                .wait_semaphores(&wait_semaphores)
                .command_buffers(&[self.command_buffers[image_index as usize]])
                .wait_dst_stage_mask(&wait_stages)
                .signal_semaphores(&single_semaphores)
                .build();

            //需要通过vkResetFences调用将其重置为手动，以将栅栏恢复为无信号状态。
            self.device
                .as_ref()
                .unwrap()
                .reset_fences(&[self.inflight_fences[self.current_frame as usize]])
                .expect("reset_fences error");

            //以传递在命令缓冲区完成执行时应发出信号的栏珊。我们可以用它来表示一帧已经结束
            self.device
                .as_ref()
                .unwrap()
                .queue_submit(
                    self.graphics_queue,
                    &[submit_info],
                    self.inflight_fences[self.current_frame as usize],
                )
                .expect("queue_submit error");

            let swap_chains = [self.swap_chain];
            let present_info = PresentInfoKHR::builder()
                .wait_semaphores(&single_semaphores)
                .swapchains(&swap_chains)
                .image_indices(&[image_index])
                .build();

            let result = self
                .swap_chain_loader
                .as_ref()
                .unwrap()
                .queue_present(self.present_queue, &present_info);

            if let Err(err) = result {
                if err == Result::ERROR_OUT_OF_DATE_KHR || err == Result::SUBOPTIMAL_KHR {
                    self.recreate_swap_chain();
                } else {
                    panic!("failed to present swap chain image!");
                }
            }

            //如果使用vkQueueWaitIdle
            //我们可能无法以这种方式最佳地使用GPU，因为整个图形流水线现在一次只能使用一帧。当前帧已经经过的阶段是空闲的，可能已经用于下一帧。
            //vkQueueWaitIdle

            //我们不应忘记每次都前进到下一帧
            self.current_frame = (self.current_frame + 1) % MAX_FRAMES_IN_FLIGHT as i32;
        }
    }

    fn update_uniform_buffer(&self, current_image: u32) {
        let proj = Perspective3::new(
            (self.swap_chain_extent.width / self.swap_chain_extent.height) as f32,
            45.0f32,
            0.1f32,
            10.0f32,
        );

        let mut proj = proj.into_inner();
        //Y翻转
        proj[(1, 1)] = proj[(1, 1)] * -1.0f32;
        let proj = Perspective3::from_matrix_unchecked(proj);

        let ubo = Box::new(UniformBufferObject {
            // foo: Vector2::new(2.0, 2.0),
            model: Matrix4::<f32>::identity(),
            view: Matrix::look_at_rh(
                &Point3::<f32>::new(2.0f32, 2.0f32, 2.0f32),
                &Point3::new(0.0f32, 0.0f32, 0.0f32),
                &Vector3::z(),
            ),
            proj,
        });

        // let a_ptr = &ubo.foo;
        // let b_ptr = &ubo.model;
        // let c_ptr = &ubo.view;

        // info!("{}", std::mem::align_of::<UniformBufferObject>());

        // let base = a_ptr as *const _ as usize;

        // println!("foo: {}", a_ptr as *const _ as usize - base);
        // println!("model: {}", b_ptr as *const _ as usize - base);
        // println!("view: {}", c_ptr as *const _ as usize - base);

        //因此我们可以将统一缓冲区对象中的数据复制到当前的统一缓冲区。发生这种情况的方式与我们使用顶点缓冲区的方式完全相同，不同之处在于没有暂存缓冲区：
        let data = unsafe {
            self.device
                .as_ref()
                .unwrap()
                .map_memory(
                    self.uniform_buffers_memory[current_image as usize],
                    0,
                    std::mem::size_of::<UniformBufferObject>() as u64,
                    MemoryMapFlags::empty(),
                )
                .expect("map_memory error") as *mut UniformBufferObject
        };

        //现在，您可以copy_from_nonoverlapping将顶点数据简单地映射到映射的内存，
        unsafe {
            let size = std::mem::size_of_val(&ubo);
            data.copy_from_nonoverlapping(&*ubo, size);

            self.device
                .as_ref()
                .unwrap()
                .unmap_memory(self.uniform_buffers_memory[current_image as usize]);
        }
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
        let device_features = PhysicalDeviceFeatures::builder()
            .sampler_anisotropy(true)
            .build();

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

        let depth_attachment = AttachmentDescription::builder()
            .format(self.find_depth_format())
            .samples(SampleCountFlags::TYPE_1)
            .load_op(AttachmentLoadOp::DONT_CARE)
            .store_op(AttachmentStoreOp::DONT_CARE)
            .stencil_load_op(AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(AttachmentStoreOp::DONT_CARE)
            .initial_layout(ImageLayout::UNDEFINED)
            .final_layout(ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
            .build();

        let depth_attachment_ref = AttachmentReference::builder()
            .attachment(1)
            .layout(ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
            .build();

        //子通道可以引用以下其他类型的附件
        //pInputAttachments：从着色器读取的附件
        //pResolveAttachments：用于多采样颜色附件的附件
        //pDepthStencilAttachment：深度和模板数据的附件
        //pPreserveAttachments：此子通道未使用的附件，但必须保留其数据

        //为第一个（也是唯一一个）子通道添加对附件的引用
        //与颜色附件不同，子通道只能使用单个深度（+模板）附件。在多个缓冲区上进行深度测试并没有任何意义。
        let sub_pass = SubpassDescription::builder()
            //Vulkan将来可能还会支持计算子通道，因此我们必须明确地说这是图形子通道
            .pipeline_bind_point(PipelineBindPoint::GRAPHICS)
            .color_attachments(&[color_attachment_ref])
            .depth_stencil_attachment(&depth_attachment_ref)
            .build();

        let mut dependency = SubpassDependency::default();
        dependency.src_subpass = SUBPASS_EXTERNAL;
        dependency.dst_subpass = 0;
        dependency.src_stage_mask = PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT;
        dependency.src_access_mask = AccessFlags::default();
        dependency.dst_stage_mask = PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT;
        dependency.dst_access_mask =
            AccessFlags::COLOR_ATTACHMENT_READ | AccessFlags::COLOR_ATTACHMENT_WRITE;

        //渲染通道
        let render_pass_info = RenderPassCreateInfo::builder()
            .attachments(&[color_attachment, depth_attachment])
            .subpasses(&[sub_pass])
            .dependencies(&[dependency])
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
    /// 我们需要提供有关着色器中用于流水线创建的每个描述符绑定的详细信息，就像我们必须对每个顶点属性及其location索引所做的一样 。
    ///
    pub(crate) fn create_descriptor_set_layout(&mut self) {
        let ubo_layout_binding = DescriptorSetLayoutBinding::builder()
            //前两个字段指定binding在着色器中使用的字段和描述符的类型
            //该描述符是一个统一的缓冲区对象
            .binding(0)
            .descriptor_type(DescriptorType::UNIFORM_BUFFER)
            //我们的MVP转换位于单个统一缓冲区对象中，因此我们使用descriptorCount的1。
            .descriptor_count(1)
            //我们还需要指定在哪个着色器阶段引用描述符
            //该stageFlags字段可以是VkShaderStageFlagBits值或value的组合VK_SHADER_STAGE_ALL_GRAPHICS
            .stage_flags(ShaderStageFlags::VERTEX)
            .build();

        //tageFlags以指示我们打算在片段着色器中使用组合的图像采样器描述符。那就是片段颜色的确定。可以在顶点着色器中使用纹理采样，例如通过heightmap使顶点网格动态变形 。
        let sampler_layout_binding = DescriptorSetLayoutBinding::builder()
            .binding(1)
            .descriptor_count(1)
            .descriptor_type(DescriptorType::COMBINED_IMAGE_SAMPLER)
            .stage_flags(ShaderStageFlags::FRAGMENT)
            .build();

        let layout_info = DescriptorSetLayoutCreateInfo::builder()
            .bindings(&[ubo_layout_binding, sampler_layout_binding])
            .build();

        self.descriptor_set_layout = unsafe {
            self.device
                .as_ref()
                .unwrap()
                .create_descriptor_set_layout(&layout_info, None)
                .expect("create_descriptor_set_layout error")
        };
    }

    ///
    /// 创建管线
    ///
    /// 可编程管线阶段
    ///
    pub(crate) fn create_graphics_pipeline(&mut self) {
        let vert_shader_code = Self::read_file(Path::new("src/shader/depth_buffering_vert.spv"));
        let frag_shader_code = Self::read_file(Path::new("src/shader/depth_buffering_frag.spv"));

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
        let mut vertex_input_info = PipelineVertexInputStateCreateInfo::default();

        let binding_description = Vertex::get_binding_description();
        let attribute_descriptions = Vertex::get_attribute_descriptions();
        vertex_input_info.vertex_binding_description_count = 1;
        vertex_input_info.vertex_attribute_description_count = attribute_descriptions.len() as u32;
        vertex_input_info.p_vertex_binding_descriptions = &binding_description;
        vertex_input_info.p_vertex_attribute_descriptions = attribute_descriptions.as_ptr();

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
            // .front_face(FrontFace::CLOCKWISE)
            //由于我们在投影矩阵中进行了Y翻转，因此现在以逆时针顺序而不是顺时针顺序绘制了顶点。这将导致背面剔除，并阻止绘制任何几何图形。
            .front_face(FrontFace::COUNTER_CLOCKWISE)
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
        let depth_stencil = PipelineDepthStencilStateCreateInfo::builder()
            .depth_test_enable(true)
            .depth_write_enable(true)
            .depth_compare_op(CompareOp::LESS)
            .depth_bounds_test_enable(false)
            .stencil_test_enable(false)
            .build();

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
        // let pipeline_layout_info = PipelineLayoutCreateInfo::builder().build();
        //我们需要在管道创建期间指定描述符集布局，以告诉Vulkan着色器将使用哪些描述符
        let pipeline_layout_info = PipelineLayoutCreateInfo::builder()
            //您可能想知道为什么在这里可以指定多个描述符集布局，因为一个已经包含了所有绑定。在下一章中，我们将介绍描述符池和描述符集。
            .set_layouts(&[self.descriptor_set_layout])
            .build();

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
            .depth_stencil_state(&depth_stencil)
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
            let attachments = vec![
                ImageView::from(swap_chain_image_view),
                self.depth_image_view,
            ];

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

    ///
    /// 创建深度图像资源
    ///
    pub(crate) fn create_depth_resources(&mut self) {
        let depth_format = self.find_depth_format();
        let (depth_image, depth_image_memory) = self.create_image(
            self.swap_chain_extent.width,
            self.swap_chain_extent.height,
            depth_format,
            ImageTiling::OPTIMAL,
            ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
            MemoryPropertyFlags::DEVICE_LOCAL,
        );

        self.depth_image = depth_image;
        self.depth_image_memory = depth_image_memory;

        self.depth_image_view =
            self.create_image_view(self.depth_image, depth_format, ImageAspectFlags::DEPTH);

        //显式过渡深度图像
        self.transition_image_layout(
            self.depth_image,
            depth_format,
            ImageLayout::UNDEFINED,
            ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
        );
    }

    ///
    /// 查找深度格式
    ///
    fn find_depth_format(&mut self) -> Format {
        self.find_support_format(
            vec![
                Format::D32_SFLOAT,
                Format::D32_SFLOAT_S8_UINT,
                Format::X8_D24_UNORM_PACK32,
            ],
            ImageTiling::OPTIMAL,
            //确保使用VK_FORMAT_FEATURE_标志而不是VK_IMAGE_USAGE_在这种情况下。所有这些候选格式均包含深度成分，但后两种还包含模板成分。
            FormatFeatureFlags::DEPTH_STENCIL_ATTACHMENT,
        )
    }

    ///
    /// 是否包含模板组件
    ///
    fn has_stencil_component(&mut self, format: Format) -> bool {
        format == Format::D32_SFLOAT_S8_UINT || format == Format::D24_UNORM_S8_UINT
    }

    ///
    /// 查找一个支持的格式
    ///
    fn find_support_format(
        &mut self,
        candidates: Vec<Format>,
        tilling: ImageTiling,
        features: FormatFeatureFlags,
    ) -> Format {
        unsafe {
            for (_i, &format) in candidates.iter().enumerate() {
                //该VkFormatProperties结构包含三个字段：

                //linearTilingFeatures：线性平铺支持的用例
                //optimalTilingFeatures：最佳平铺支持的用例
                //bufferFeatures：缓冲区支持的用例
                let props: FormatProperties = self
                    .instance
                    .as_ref()
                    .unwrap()
                    .get_physical_device_format_properties(self.physical_device.unwrap(), format);

                if tilling == ImageTiling::LINEAR
                    && (props.linear_tiling_features & features) == features
                {
                    return format;
                } else if tilling == ImageTiling::OPTIMAL
                    && (props.optimal_tiling_features & features) == features
                {
                    return format;
                }
            }

            panic!("failed to find supported format!");
        };
    }

    ///
    /// 创建纹理
    /// 在该函数中将加载图像并将其上传到Vulkan图像对象中。
    ///
    pub(crate) fn create_texture_image(&mut self) {
        let img = image::open(TEXTURE_PATH).unwrap();
        let (tex_width, tex_height) = img.dimensions();
        let image_size =
            (std::mem::size_of::<u8>() as u32 * tex_width * tex_height * 4) as DeviceSize;

        let (staging_buffer, staging_buffer_memory) = self.create_buffer(
            image_size,
            BufferUsageFlags::TRANSFER_SRC,
            MemoryPropertyFlags::HOST_VISIBLE | MemoryPropertyFlags::HOST_COHERENT,
        );

        let data = unsafe {
            self.device
                .as_ref()
                .unwrap()
                .map_memory(
                    staging_buffer_memory,
                    0,
                    image_size,
                    MemoryMapFlags::empty(),
                )
                .expect("map_memory error") as *mut u8
        };

        unsafe {
            let pixels = img.to_rgba().into_raw();
            data.copy_from_nonoverlapping(pixels.as_ptr(), pixels.len());

            self.device
                .as_ref()
                .unwrap()
                .unmap_memory(staging_buffer_memory);
        }

        let (texture_image, texture_image_memory) = self.create_image(
            tex_width,
            tex_height,
            Format::R8G8B8A8_SRGB,
            ImageTiling::OPTIMAL,
            ImageUsageFlags::TRANSFER_DST | ImageUsageFlags::SAMPLED,
            MemoryPropertyFlags::DEVICE_LOCAL,
        );

        self.texture_image = texture_image;
        self.textre_image_memory = texture_image_memory;

        self.transition_image_layout(
            self.texture_image,
            Format::R8G8B8A8_SRGB,
            ImageLayout::UNDEFINED,
            ImageLayout::TRANSFER_DST_OPTIMAL,
        );
        self.copy_buffer_to_image(
            staging_buffer,
            self.texture_image,
            tex_width as u32,
            tex_height as u32,
        );
        self.transition_image_layout(
            self.texture_image,
            Format::R8G8B8A8_SRGB,
            ImageLayout::TRANSFER_DST_OPTIMAL,
            ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        );

        unsafe {
            self.device
                .as_ref()
                .unwrap()
                .destroy_buffer(staging_buffer, None);
            self.device
                .as_ref()
                .unwrap()
                .free_memory(staging_buffer_memory, None);
        };
    }

    fn create_texture_image_view(&mut self) {
        self.texture_image_view = self.create_image_view(
            self.texture_image,
            Format::R8G8B8A8_SRGB,
            ImageAspectFlags::COLOR,
        );
    }

    fn create_texture_sampler(&mut self) {
        let sampler_info = SamplerCreateInfo::builder()
            .mag_filter(Filter::LINEAR)
            .min_filter(Filter::LINEAR)
            .address_mode_u(SamplerAddressMode::REPEAT)
            .address_mode_v(SamplerAddressMode::REPEAT)
            .address_mode_w(SamplerAddressMode::REPEAT)
            .anisotropy_enable(true)
            .max_anisotropy(16.0f32)
            .border_color(BorderColor::INT_OPAQUE_BLACK)
            .unnormalized_coordinates(false)
            .compare_enable(false)
            .compare_op(CompareOp::ALWAYS)
            .mipmap_mode(SamplerMipmapMode::LINEAR)
            .mip_lod_bias(0.0f32)
            .min_lod(0.0f32)
            .max_lod(0.0f32)
            .build();

        //请注意，采样器未引用VkImage任何地方。采样器是一个独特的对象，提供了从纹理中提取颜色的接口。它可以应用于所需的任何图像，无论是1D，2D还是3D。这与许多较早的API不同，后者将纹理图像和过滤合并为一个状态。

        self.texture_sampler = unsafe {
            self.device
                .as_ref()
                .unwrap()
                .create_sampler(&sampler_info, None)
                .expect("create_sampler error")
        };
    }

    fn create_image_view(
        &mut self,
        image: Image,
        format: Format,
        aspect_mask: ImageAspectFlags,
    ) -> ImageView {
        let view_info = ImageViewCreateInfo::builder()
            .image(image)
            .view_type(ImageViewType::TYPE_2D)
            .format(format)
            .subresource_range(ImageSubresourceRange {
                aspect_mask: aspect_mask,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            })
            .build();

        let texture_image_view = unsafe {
            self.device
                .as_ref()
                .unwrap()
                .create_image_view(&view_info, None)
                .expect("create_image_view error")
        };

        texture_image_view
    }

    fn begin_single_time_commands(&self) -> CommandBuffer {
        let alloc_info = CommandBufferAllocateInfo::builder()
            .level(CommandBufferLevel::PRIMARY)
            .command_pool(self.command_pool)
            .command_buffer_count(1)
            .build();

        let command_buffer = unsafe {
            self.device
                .as_ref()
                .unwrap()
                .allocate_command_buffers(&alloc_info)
                .expect("allocate_command_buffers error")[0]
        };

        //并立即开始记录命令缓冲区
        let begin_info = CommandBufferBeginInfo::builder()
            // 我们将只使用一次命令缓冲区，然后等待从函数返回，直到复制操作完成执行。最好告诉我们使用的意图VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT。
            .flags(CommandBufferUsageFlags::ONE_TIME_SUBMIT)
            .build();

        unsafe {
            self.device
                .as_ref()
                .unwrap()
                .begin_command_buffer(command_buffer, &begin_info)
                .expect("begin_command_buffer error")
        };

        command_buffer
    }

    fn end_single_time_commands(&mut self, command_buffer: CommandBuffer) {
        unsafe {
            //该命令缓冲区仅包含复制命令，因此我们可以在此之后立即停止记录。
            self.device
                .as_ref()
                .unwrap()
                .end_command_buffer(command_buffer)
                .expect("end_command_buffer error");

            let submit_info = SubmitInfo::builder()
                .command_buffers(&[command_buffer])
                .build();

            //与绘制命令不同，这次没有任何事件需要等待。我们只想立即在缓冲区上执行传输。还有两种可能的方法来等待此传输完成。我们可以使用栅栏等待vkWaitForFences，或者只是等待传输队列变得空闲vkQueueWaitIdle。栅栏允许您同时安排多个传输并等待所有传输完成，而不必一次执行一个。这可以为程序员提供更多优化的机会。
            self.device
                .as_ref()
                .unwrap()
                .queue_submit(self.graphics_queue, &[submit_info], Fence::null())
                .expect("queue_submit error");

            self.device
                .as_ref()
                .unwrap()
                .queue_wait_idle(self.graphics_queue)
                .expect("queue_wait_idle error");

            //不要忘记清理用于传输操作的命令缓冲区。
            self.device
                .as_ref()
                .unwrap()
                .free_command_buffers(self.command_pool, &[command_buffer]);
        };
    }

    ///
    /// 如果我们仍在使用缓冲区，那么我们现在可以编写一个函数来记录并执行vkCmdCopyBufferToImage以完成作业，但是此命令要求图像首先位于正确的布局中。创建一个新函数来处理布局转换：
    ///
    fn transition_image_layout(
        &mut self,
        image: Image,
        format: Format,
        old_layout: ImageLayout,
        new_layout: ImageLayout,
    ) {
        let command_buffer = self.begin_single_time_commands();

        //执行布局转换的最常见方法之一是使用图像存储屏障。像这样的流水线屏障通常用于同步对资源的访问，例如确保对缓冲区的写入在从缓冲区中读取之前完成，但是它也可以用于转换图像布局并在VK_SHARING_MODE_EXCLUSIVE使用时转移队列族所有权。对于缓冲区，存在等效的缓冲区内存屏障。

        let mut barrier = ImageMemoryBarrier::builder()
            .old_layout(old_layout)
            .new_layout(new_layout)
            .src_queue_family_index(QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(QUEUE_FAMILY_IGNORED)
            .image(image)
            .subresource_range(ImageSubresourceRange {
                aspect_mask: ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            })
            .build();

        if new_layout == ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL {
            barrier.subresource_range.aspect_mask = ImageAspectFlags::DEPTH;

            if self.has_stencil_component(format) {
                barrier.subresource_range.aspect_mask |= ImageAspectFlags::STENCIL;
            }
        }

        let (source_stage, destination_stage) = {
            if old_layout == ImageLayout::UNDEFINED
                && new_layout == ImageLayout::TRANSFER_DST_OPTIMAL
            {
                barrier.src_access_mask = AccessFlags::empty();
                barrier.dst_access_mask = AccessFlags::TRANSFER_WRITE;

                (
                    PipelineStageFlags::TOP_OF_PIPE,
                    PipelineStageFlags::TRANSFER,
                )
            } else if old_layout == ImageLayout::TRANSFER_DST_OPTIMAL
                && new_layout == ImageLayout::SHADER_READ_ONLY_OPTIMAL
            {
                barrier.src_access_mask = AccessFlags::TRANSFER_WRITE;
                barrier.dst_access_mask = AccessFlags::SHADER_READ;

                (
                    PipelineStageFlags::TRANSFER,
                    PipelineStageFlags::FRAGMENT_SHADER,
                )
            } else if old_layout == ImageLayout::UNDEFINED
                && new_layout == ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL
            {
                barrier.src_access_mask = AccessFlags::empty();
                barrier.dst_access_mask = AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ
                    | AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE;

                (
                    PipelineStageFlags::TOP_OF_PIPE,
                    PipelineStageFlags::EARLY_FRAGMENT_TESTS,
                )
            } else {
                panic!("unsupported layout transition!");
            }
        };

        unsafe {
            self.device.as_ref().unwrap().cmd_pipeline_barrier(
                command_buffer,
                source_stage,
                destination_stage,
                DependencyFlags::empty(),
                &[],
                &[],
                &[barrier],
            );
        };

        self.end_single_time_commands(command_buffer);
    }

    pub(crate) fn copy_buffer_to_image(
        &mut self,
        buffer: Buffer,
        image: Image,
        width: u32,
        height: u32,
    ) {
        let command_buffer = self.begin_single_time_commands();

        let region = BufferImageCopy::builder()
            .buffer_row_length(0)
            .buffer_image_height(0)
            .image_subresource(ImageSubresourceLayers {
                aspect_mask: ImageAspectFlags::COLOR,
                mip_level: 0,
                base_array_layer: 0,
                layer_count: 1,
            })
            .image_offset(Offset3D { x: 0, y: 0, z: 0 })
            .image_extent(Extent3D {
                width,
                height,
                depth: 1,
            })
            .build();

        unsafe {
            self.device.as_ref().unwrap().cmd_copy_buffer_to_image(
                command_buffer,
                buffer,
                image,
                ImageLayout::TRANSFER_DST_OPTIMAL,
                &[region],
            );
        };

        self.end_single_time_commands(command_buffer);
    }

    ///
    /// 因为我们将在本章中创建多个缓冲区，所以将缓冲区创建移至辅助函数是一个好主意。
    ///
    pub(crate) fn create_buffer(
        &self,
        size: DeviceSize,
        usage: BufferUsageFlags,
        properties: MemoryPropertyFlags,
    ) -> (Buffer, DeviceMemory) {
        let buffer_info = BufferCreateInfo::builder()
            .size(size)
            .usage(usage)
            .sharing_mode(SharingMode::EXCLUSIVE)
            .build();

        let buffer = unsafe {
            self.device
                .as_ref()
                .unwrap()
                .create_buffer(&buffer_info, None)
                .expect("create_buffer error")
        };

        //内存需求
        //缓冲区已创建，但实际上尚未分配任何内存。为缓冲区分配内存
        //该VkMemoryRequirements结构具有三个字段：
        //size：所需的内存量（以字节为单位）可能与有所不同 bufferInfo.size。
        //alignment：缓冲区从内存分配的区域开始的偏移量（以字节为单位）取决于bufferInfo.usage和bufferInfo.flags。
        //memoryTypeBits：适用于缓冲区的内存类型的位字段。
        //
        let mem_requirements: MemoryRequirements = unsafe {
            self.device
                .as_ref()
                .unwrap()
                .get_buffer_memory_requirements(buffer)
        };

        //内存分配
        let alloc_info = MemoryAllocateInfo::builder()
            .allocation_size(mem_requirements.size)
            .memory_type_index(self.find_memory_type(
                mem_requirements.memory_type_bits,
                //不幸的是，例如由于缓存，驱动程序可能不会立即将数据复制到缓冲存储器中。也有可能在映射的内存中尚不可见对缓冲区的写入。有两种方法可以解决该问题：
                //使用主机一致的内存堆，用 VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
                //打电话vkFlushMappedMemoryRanges到写入内存映射，并调用后vkInvalidateMappedMemoryRanges从映射内存读取前

                //我们采用第一种方法，该方法可确保映射的内存始终与分配的内存的内容匹配。请记住，与显式刷新相比，这可能会导致性能稍差，但是我们将在下一章中了解为什么这无关紧要。
                properties,
            ));

        let buffer_memory = unsafe {
            self.device
                .as_ref()
                .unwrap()
                .allocate_memory(&alloc_info, None)
                .expect("allocate_memory error")
        };

        //如果内存分配成功，那么我们现在可以使用将该内存与缓冲区关联
        unsafe {
            self.device
                .as_ref()
                .unwrap()
                //由于此内存是专门为此顶点缓冲区分配的，因此偏移量仅为0。如果偏移量不为零，则需要被整除memRequirements.alignment。
                .bind_buffer_memory(buffer, buffer_memory, 0)
                .expect("bind_buffer_memory error");
        };

        (buffer, buffer_memory)
    }

    pub(crate) fn load_model(&mut self) {
        let cornell_box = tobj::load_obj(&Path::new(MODEL_PATH));
        assert!(cornell_box.is_ok());
        let (models, _materials) = cornell_box.unwrap();

        for (_i, m) in models.iter().enumerate() {
            let mesh = &m.mesh as &tobj::Mesh;

            for v in 0..mesh.positions.len() / 3 {
                let vertex = Vertex {
                    pos: Vector3::<f32>::new(
                        mesh.positions[v * 3],
                        mesh.positions[v * 3 + 1],
                        mesh.positions[v * 3 + 2],
                    ),
                    color: Vector3::<f32>::new(1.0, 1.0, 1.0),
                    text_coord: Vector2::<f32>::new(
                        mesh.texcoords[v * 2],
                        mesh.texcoords[v * 2 + 1],
                    ),
                };

                self.vertices.push(vertex);
                self.indices.push(self.indices.len() as u32);
            }
        }
    }

    ///
    /// 缓冲区创建
    ///
    /// 使用staging_buffer
    ///
    /// 运行程序以确认您再次看到了熟悉的三角形。这种改进可能暂时不可见，但是现在它的顶点数据正在从高性能内存中加载。当我们要开始渲染更复杂的几何图形时，这将很重要。
    ///
    ///
    pub(crate) fn create_vertex_buffer(&mut self) {
        // let vertices = self.vertices; // unsafe { &*VERTICES };
        let buffer_size =
            std::mem::size_of_val(&self.vertices[0]) as u64 * self.vertices.len() as u64;

        //我们现在使用一个新的stagingBufferwith stagingBufferMemory来映射和复制顶点数据。
        let (staging_buffer, staging_buffer_memory) = self.create_buffer(
            buffer_size,
            //BUFFER_USAGE_TRANSFER_SRC_BIT：缓冲区可用作存储器传输操作中的源。
            //BUFFER_USAGE_TRANSFER_DST_BIT：缓冲区可以在存储器传输操作中用作目标。
            BufferUsageFlags::TRANSFER_SRC,
            MemoryPropertyFlags::HOST_VISIBLE | MemoryPropertyFlags::HOST_COHERENT,
        );

        //填充顶点缓冲区
        let data = unsafe {
            self.device
                .as_ref()
                .unwrap()
                .map_memory(
                    staging_buffer_memory,
                    0,
                    buffer_size,
                    MemoryMapFlags::empty(),
                )
                .expect("map_memory error") as *mut Vertex
        };

        //现在，您可以copy_from_nonoverlapping将顶点数据简单地映射到映射的内存，
        unsafe {
            data.copy_from_nonoverlapping(self.vertices.as_ptr(), self.vertices.len());

            //不幸的是，例如由于缓存，驱动程序可能不会立即将数据复制到缓冲存储器中。也有可能在映射的内存中尚不可见对缓冲区的写入。有两种方法可以解决该问题：
            //使用主机一致的内存堆，用 VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
            //打电话vkFlushMappedMemoryRanges到写入内存映射，并调用后vkInvalidateMappedMemoryRanges从映射内存读取前

            //我们采用第一种方法，该方法可确保映射的内存始终与分配的内存的内容匹配。请记住，与显式刷新相比，这可能会导致性能稍差，但是我们将在下一章中了解为什么这无关紧要。

            //刷新内存范围或使用相关的内存堆意味着驱动程序将意识到我们对缓冲区的写入，但是这并不意味着它们实际上在GPU上是可见的。数据传输到GPU的操作是在后台进行的，

            //使用再次取消映射vkUnmapMemory
            self.device
                .as_ref()
                .unwrap()
                .unmap_memory(staging_buffer_memory);

            let (vertex_buffer, vertex_buffer_memory) = self.create_buffer(
                buffer_size,
                BufferUsageFlags::TRANSFER_DST | BufferUsageFlags::VERTEX_BUFFER,
                MemoryPropertyFlags::DEVICE_LOCAL,
            );

            self.vertex_buffer = vertex_buffer;
            self.vertex_buffer_memory = vertex_buffer_memory;

            self.copy_buffer(staging_buffer, self.vertex_buffer, buffer_size);

            self.device
                .as_ref()
                .unwrap()
                .destroy_buffer(staging_buffer, None);

            self.device
                .as_ref()
                .unwrap()
                .free_memory(staging_buffer_memory, None);
        }
    }

    ///
    /// 创建索引缓冲区
    ///
    pub(crate) fn create_index_buffer(&mut self) {
        let buffer_size =
            std::mem::size_of_val(&self.indices[0]) as u64 * self.indices.len() as u64;

        //我们现在使用一个新的stagingBufferwith stagingBufferMemory来映射和复制顶点数据。
        let (staging_buffer, staging_buffer_memory) = self.create_buffer(
            buffer_size,
            //BUFFER_USAGE_TRANSFER_SRC_BIT：缓冲区可用作存储器传输操作中的源。
            //BUFFER_USAGE_TRANSFER_DST_BIT：缓冲区可以在存储器传输操作中用作目标。
            BufferUsageFlags::TRANSFER_SRC,
            MemoryPropertyFlags::HOST_VISIBLE | MemoryPropertyFlags::HOST_COHERENT,
        );

        //填充顶点缓冲区
        let data = unsafe {
            self.device
                .as_ref()
                .unwrap()
                .map_memory(
                    staging_buffer_memory,
                    0,
                    buffer_size,
                    MemoryMapFlags::empty(),
                )
                .expect("map_memory error") as *mut u32
        };

        //现在，您可以copy_from_nonoverlapping将顶点数据简单地映射到映射的内存，
        unsafe {
            data.copy_from_nonoverlapping(self.indices.as_ptr(), self.indices.len());

            //不幸的是，例如由于缓存，驱动程序可能不会立即将数据复制到缓冲存储器中。也有可能在映射的内存中尚不可见对缓冲区的写入。有两种方法可以解决该问题：
            //使用主机一致的内存堆，用 VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
            //打电话vkFlushMappedMemoryRanges到写入内存映射，并调用后vkInvalidateMappedMemoryRanges从映射内存读取前

            //我们采用第一种方法，该方法可确保映射的内存始终与分配的内存的内容匹配。请记住，与显式刷新相比，这可能会导致性能稍差，但是我们将在下一章中了解为什么这无关紧要。

            //刷新内存范围或使用相关的内存堆意味着驱动程序将意识到我们对缓冲区的写入，但是这并不意味着它们实际上在GPU上是可见的。数据传输到GPU的操作是在后台进行的，

            //使用再次取消映射vkUnmapMemory
            self.device
                .as_ref()
                .unwrap()
                .unmap_memory(staging_buffer_memory);

            let (index_buffer, index_buffer_memory) = self.create_buffer(
                buffer_size,
                BufferUsageFlags::TRANSFER_DST | BufferUsageFlags::INDEX_BUFFER,
                MemoryPropertyFlags::DEVICE_LOCAL,
            );

            self.index_buffer = index_buffer;
            self.index_buffer_memory = index_buffer_memory;

            self.copy_buffer(staging_buffer, self.index_buffer, buffer_size);

            self.device
                .as_ref()
                .unwrap()
                .destroy_buffer(staging_buffer, None);

            self.device
                .as_ref()
                .unwrap()
                .free_memory(staging_buffer_memory, None);
        }
    }

    ///
    /// 创建统一缓冲区(UBO)
    ///
    pub(crate) fn create_uniform_buffers(&mut self) {
        let buffer_size = std::mem::size_of::<UniformBufferObject>() as DeviceSize;
        self.uniform_buffers = Vec::with_capacity(self.swap_chain_images.len());
        self.uniform_buffers_memory = Vec::with_capacity(self.swap_chain_images.len());

        for _i in 0..self.swap_chain_images.len() {
            let (uniform_buffer, uniform_buffer_memory) = self.create_buffer(
                buffer_size,
                BufferUsageFlags::UNIFORM_BUFFER,
                MemoryPropertyFlags::HOST_VISIBLE | MemoryPropertyFlags::HOST_COHERENT,
            );

            self.uniform_buffers.push(uniform_buffer);
            self.uniform_buffers_memory.push(uniform_buffer_memory);
        }
    }

    ///
    /// 构建描述符池
    ///
    pub(crate) fn create_descriptor_pool(&mut self) {
        //我们首先需要使用VkDescriptorPoolSize结构来描述我们的描述符集将包含哪些描述符类型以及其中有多少个描述符类型。
        let ubo_pool_size = DescriptorPoolSize::builder()
            .ty(DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(self.swap_chain_images.len() as u32)
            .build();

        // 纹理
        let img_pool_size = DescriptorPoolSize::builder()
            .ty(DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(self.swap_chain_images.len() as u32)
            .build();

        let pool_info = DescriptorPoolCreateInfo::builder()
            .pool_sizes(&[ubo_pool_size, img_pool_size])
            //除了可用的单个描述符的最大数量外，我们还需要指定可以分配的最大描述符集数量：
            .max_sets(self.swap_chain_images.len() as u32)
            .build();

        self.descriptor_pool = unsafe {
            self.device
                .as_ref()
                .unwrap()
                .create_descriptor_pool(&pool_info, None)
                .expect("create_descriptor_pool error")
        };
    }

    ///
    /// 构建描述符集
    ///
    pub(crate) fn create_descriptor_sets(&mut self) {
        let mut layouts = Vec::<DescriptorSetLayout>::with_capacity(self.swap_chain_images.len());
        for _i in 0..self.swap_chain_images.len() {
            layouts.push(self.descriptor_set_layout);
        }

        let alloc_info = DescriptorSetAllocateInfo::builder()
            .descriptor_pool(self.descriptor_pool)
            .set_layouts(&layouts)
            .build();

        //您不需要显式清理描述符集，因为在销毁描述符池时，它们将自动释放。
        self.descriptor_sets = unsafe {
            self.device
                .as_ref()
                .unwrap()
                .allocate_descriptor_sets(&alloc_info)
                .expect("allocate_descriptor_sets error")
        };

        //现在已经分配了描述符集，但是仍然需要配置其中的描述符
        for i in 0..self.descriptor_sets.len() {
            let buffer_info = DescriptorBufferInfo::builder()
                .buffer(self.uniform_buffers[i])
                .offset(0)
                .range(std::mem::size_of::<UniformBufferObject>() as u64)
                .build();

            let image_info = DescriptorImageInfo::builder()
                .image_layout(ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .image_view(self.texture_image_view)
                .sampler(self.texture_sampler)
                .build();

            let descriptor_write = WriteDescriptorSet::builder()
                .dst_set(self.descriptor_sets[i])
                .dst_binding(0)
                .dst_array_element(0)
                .descriptor_type(DescriptorType::UNIFORM_BUFFER)
                .buffer_info(&[buffer_info])
                .build();

            let descriptor_write1 = WriteDescriptorSet::builder()
                .dst_set(self.descriptor_sets[i])
                .dst_binding(1)
                .dst_array_element(0)
                .descriptor_type(DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(&[image_info])
                .build();

            unsafe {
                self.device
                    .as_ref()
                    .unwrap()
                    .update_descriptor_sets(&[descriptor_write, descriptor_write1], &[]);
            };
        }
    }

    ///
    /// 缓冲区拷贝
    ///
    pub(crate) fn copy_buffer(&mut self, src_buffer: Buffer, dst_buffer: Buffer, size: DeviceSize) {
        let command_buffer = self.begin_single_time_commands();

        let copy_region = BufferCopy::builder().size(size).build();
        unsafe {
            //使用该vkCmdCopyBuffer命令传输缓冲区的内容。它以源缓冲区和目标缓冲区作为参数，并复制一个区域数组。区域以VkBufferCopy结构定义，由源缓冲区偏移量，目标缓冲区偏移量和大小组成。
            self.device.as_ref().unwrap().cmd_copy_buffer(
                command_buffer,
                src_buffer,
                dst_buffer,
                &[copy_region],
            );
        };

        self.end_single_time_commands(command_buffer);
    }

    pub(crate) fn create_image(
        &mut self,
        width: u32,
        height: u32,
        format: Format,
        tiling: ImageTiling,
        usage: ImageUsageFlags,
        properties: MemoryPropertyFlags,
    ) -> (Image, DeviceMemory) {
        let image_info = ImageCreateInfo::builder()
            //在该imageType字段中指定的图像类型告诉Vulkan，图像中的纹素将要处理哪种坐标系。可以创建1D，2D和3D图像
            //例如，一维图像可用于存储数据或渐变的数组，二维图像主要用于纹理，而三维图像可用于存储体素体积。
            .image_type(ImageType::TYPE_2D)
            //指定图像的尺寸，基本上是每个轴上有多少个纹​​理像素。这就是为什么depth必须1代替0的原因
            .extent(Extent3D {
                width,
                height,
                depth: 1,
            })
            .mip_levels(1)
            .array_layers(1)
            .format(format)
            //该tiling字段可以具有两个值之一：
            //IMAGE_TILING_LINEAR：像我们的pixels数组一样，以行优先顺序排列像素
            //IMAGE_TILING_OPTIMAL：以实现定义的顺序排列Texel，以实现最佳访问
            .tiling(tiling)
            //_IMAGE_LAYOUT_UNDEFINED：GPU无法使用，并且第一个过渡将丢弃纹理像素。
            //IMAGE_LAYOUT_PREINITIALIZED：GPU无法使用，但第一个过渡将保留纹理像素。
            .initial_layout(ImageLayout::UNDEFINED)
            .usage(usage)
            .samples(SampleCountFlags::TYPE_1)
            .sharing_mode(SharingMode::EXCLUSIVE)
            .build();

        let image = unsafe {
            self.device
                .as_ref()
                .unwrap()
                .create_image(&image_info, None)
                .expect("create_image error")
        };

        let mem_requirements = unsafe {
            self.device
                .as_ref()
                .unwrap()
                .get_image_memory_requirements(image)
        };

        let alloc_info = MemoryAllocateInfo::builder()
            .allocation_size(mem_requirements.size)
            .memory_type_index(self.find_memory_type(mem_requirements.memory_type_bits, properties))
            .build();

        let image_memory = unsafe {
            self.device
                .as_ref()
                .unwrap()
                .allocate_memory(&alloc_info, None)
                .expect("allocate_memory error")
        };

        unsafe {
            self.device
                .as_ref()
                .unwrap()
                .bind_image_memory(image, image_memory, 0)
                .expect("bind_image_memory error")
        };

        (image, image_memory)
    }

    ///
    /// 图形卡可以提供不同类型的内存以进行分配。每种类型的内存在允许的操作和性能特征方面都不同。我们需要结合缓冲区的要求和我们自己的应用程序要求来找到要使用的正确类型的内存
    /// 该typeFilter参数将用于指定合适的存储器类型的位字段。这意味着我们可以通过简单地遍历它们并检查相应的位是否设置为来找到合适的内存类型的索引1。
    ///
    pub(crate) fn find_memory_type(
        &self,
        type_filter: u32,
        properties: MemoryPropertyFlags,
    ) -> u32 {
        //我们需要使用来查询有关可用内存类型的信息
        //结构具有两个数组memoryTypes 和memoryHeaps。内存堆是不同的内存资源，例如专用VRAM和RAM中的交换空间（当VRAM用完时）。这些堆中存在不同类型的内存。
        //现在，我们只关心内存的类型，而不关心它来自的堆，但是您可以想象这会影响性能。
        let mem_properties: PhysicalDeviceMemoryProperties = unsafe {
            self.instance
                .as_ref()
                .unwrap()
                .get_physical_device_memory_properties(self.physical_device.unwrap())
        };

        //首先让我们找到适合缓冲区本身的内存类型：
        for i in 0..mem_properties.memory_type_count {
            //我们不仅对适用于顶点缓冲区的内存类型感兴趣。我们还需要能够将顶点数据写入该内存。该memoryTypes数组由VkMemoryType指定每种类型的内存的堆和属性的结构组成。这些属性定义了内存的特殊功能，例如能够对其进行映射，以便我们可以从CPU对其进行写入。该属性用表示VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT，但我们也需要使用该VK_MEMORY_PROPERTY_HOST_COHERENT_BIT属性。我们将在映射内存时看到原因。
            if type_filter & (1 << i) > 0
                 //也检查此属性的支持：
                && mem_properties.memory_types[i as usize]
                    .property_flags
                    .contains(properties)
            {
                return i;
            }
        }

        panic!("failed to find suitable memory type!");
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

                let clear_values = [
                    ClearValue {
                        color: ClearColorValue {
                            float32: [0.0, 0.0, 0.0, 1.0],
                        },
                    },
                    ClearValue {
                        //在深度缓冲器深度范围是0.0对1.0，其中1.0 位于在远视点平面和0.0在近视点平面。深度缓冲区中每个点的初始值应为最远的深度，即1.0。
                        depth_stencil: ClearDepthStencilValue {
                            depth: 1.0f32,
                            stencil: 0,
                        },
                    },
                ];

                //开始渲染过程
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

                //绑定顶点缓冲区
                let vertex_buffers = [self.vertex_buffer];
                let offsets = [0];

                self.device.as_ref().unwrap().cmd_bind_vertex_buffers(
                    self.command_buffers[i],
                    0,
                    &vertex_buffers,
                    &offsets,
                );

                self.device.as_ref().unwrap().cmd_bind_index_buffer(
                    self.command_buffers[i],
                    self.index_buffer,
                    0,
                    //其中的字节偏移量以及索引数据的类型作为参数。如前所述，可能的类型为VK_INDEX_TYPE_UINT16和 VK_INDEX_TYPE_UINT32。
                    IndexType::UINT32,
                );

                //现在，我们需要更新createCommandBuffers函数，以使用实际将每个交换链图像的正确描述符集绑定到着色器中的描述符cmdBindDescriptorSets。这需要在vkCmdDrawIndexed调用之前完成：
                self.device.as_ref().unwrap().cmd_bind_descriptor_sets(
                    self.command_buffers[i],
                    PipelineBindPoint::GRAPHICS,
                    self.pipeline_layout,
                    0,
                    &[self.descriptor_sets[i]],
                    &[],
                );

                // 使用索引缓冲区后修改cmd_draw为cmd_draw_indexed
                // self.device
                //     .as_ref()
                //     .unwrap()
                //     .cmd_draw(command_buffer, 3, 1, 0, 0);

                self.device.as_ref().unwrap().cmd_draw_indexed(
                    self.command_buffers[i],
                    self.indices.len() as u32,
                    1,
                    0,
                    0,
                    0,
                );

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
    /// 创建信号量
    ///
    pub(crate) fn create_sync_objects(&mut self) {
        self.image_available_semaphores = Vec::with_capacity(MAX_FRAMES_IN_FLIGHT);
        self.render_finished_semaphores = Vec::with_capacity(MAX_FRAMES_IN_FLIGHT);
        self.inflight_fences = Vec::with_capacity(MAX_FRAMES_IN_FLIGHT);

        //没有一个框架在使用图像，因此我们将其显式初始化为没有阑珊。
        for _i in 0..self.swap_chain_images.len() {
            self.images_inflight.push(Fence::null());
        }

        let semaphore_info = SemaphoreCreateInfo::builder().build();

        //栏珊是在无信号状态下创建的。这意味着vkWaitForFences如果我们之前没有使用过围栏，它将永远等待。为了
        //解决这个问题，我们可以更改栏珊的创建，使其以信号状态进行初始化，就好像我们渲染了完成的初始帧一样：
        let fence_info = FenceCreateInfo::builder()
            .flags(FenceCreateFlags::SIGNALED)
            .build();

        unsafe {
            for _i in 0..MAX_FRAMES_IN_FLIGHT {
                self.image_available_semaphores.push(
                    self.device
                        .as_ref()
                        .unwrap()
                        .create_semaphore(&semaphore_info, None)
                        .expect("create_semaphore error"),
                );

                self.render_finished_semaphores.push(
                    self.device
                        .as_ref()
                        .unwrap()
                        .create_semaphore(&semaphore_info, None)
                        .expect("create_semaphore error"),
                );

                self.inflight_fences.push(
                    self.device
                        .as_ref()
                        .unwrap()
                        .create_fence(&fence_info, None)
                        .expect("create_fence error"),
                );
            }
        };
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

        let supported_features: PhysicalDeviceFeatures = unsafe {
            self.instance
                .as_ref()
                .unwrap()
                .get_physical_device_features(*device)
        };

        let mut swap_chain_adequate = false;
        if extensions_supported {
            let swap_chain_support = self.query_swap_chain_support(device);
            swap_chain_adequate = !swap_chain_support.formats.is_empty()
                && !swap_chain_support.present_modes.is_empty();
        }

        return indices.is_complete()
            && extensions_supported
            && swap_chain_adequate
            && supported_features.sampler_anisotropy == 1;
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
        // info!("物理设备属性：{:#?}", physical_device_properties);

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

            self.cleanup_swap_chain();

            let device = self.device.as_ref().unwrap();

            device.destroy_image_view(self.depth_image_view, None);

            device.destroy_image(self.depth_image, None);

            device.free_memory(self.depth_image_memory, None);

            device.destroy_sampler(self.texture_sampler, None);

            device.destroy_image_view(self.texture_image_view, None);

            device.destroy_image(self.texture_image, None);

            device.free_memory(self.textre_image_memory, None);

            device.destroy_descriptor_set_layout(self.descriptor_set_layout, None);

            device.destroy_buffer(self.vertex_buffer, None);

            device.free_memory(self.vertex_buffer_memory, None);

            device.destroy_buffer(self.index_buffer, None);

            device.free_memory(self.index_buffer_memory, None);

            if let Some(instance) = self.instance.as_ref() {
                for i in 0..MAX_FRAMES_IN_FLIGHT {
                    device.destroy_semaphore(self.image_available_semaphores[i], None);

                    device.destroy_semaphore(self.render_finished_semaphores[i], None);

                    device.destroy_fence(self.inflight_fences[i], None);
                }

                device.destroy_command_pool(self.command_pool, None);

                if let Some(device) = self.device.as_ref() {
                    device.destroy_device(None);
                }

                if let Some(surface_khr) = self.surface {
                    self.surface_loader
                        .as_ref()
                        .unwrap()
                        .destroy_surface(surface_khr, None);
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
        unsafe {
            //其中的所有操作drawFrame都是异步的。这意味着当我们退出中的循环时mainLoop，绘图和演示操作可能仍在进行。
            //在发生这种情况时清理资源是一个坏主意。要解决该问题，我们应该等待逻辑设备完成操作
            self.device
                .as_ref()
                .unwrap()
                .device_wait_idle()
                .expect("device_wait_idle error");
        }
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
/// 我们可以为每个顶点将任意属性传递给顶点着色器，但是全局变量呢？从本章开始，我们将继续介绍3D图形，这需要一个模型-视图-投影矩阵。我们可以将其包含为顶点数据，但这会浪费内存，并且每当转换发生更改时，都需要我们更新顶点缓冲区。转换可以轻松地更改每个帧。
/// 在Vulkan中解决此问题的正确方法是使用资源描述符。描述符是着色器自由访问缓冲区和图像等资源的一种方式。我们将建立一个包含转换矩阵的缓冲区，并让顶点着色器通过描述符访问它们。描述符的使用包括三个部分：
/// 在管道创建期间指定描述符布局
/// 从描述符池分配描述符集
/// 在渲染期间绑定描述符集
/// 该描述符布局指定资源是要由管道访问的类型，就像一个渲染通道指定将被访问的附件的类型。甲描述符组指定将绑定到描述符的实际缓冲器或图像资源，就像一个帧缓冲器指定的实际图像视图绑定到渲染过程的附件。然后将描述符集绑定到绘制命令，就像顶点缓冲区和帧缓冲区一样。
/// 描述符的类型很多，但是在本章中，我们将使用统一缓冲区对象（UBO）。在以后的章节中，我们将介绍其他类型的描述符，但是基本过程是相同的。
///
/// 已使用每顶点颜色为几何图形着色，这是一种相当有限的方法。在本教程的这一部分中，我们将实现纹理映射以使几何看起来更有趣。这也将使我们在以后的章节中加载和绘制基本的3D模型。
///
/// 向我们的应用程序添加纹理将涉及以下步骤：
/// 创建由设备内存支持的图像对象
/// 用图像文件中的像素填充
/// 创建一个图像采样器
/// 添加组合的图像采样器描述符以从纹理中采样颜色
///
/// 我们之前已经使用过图像对象，但是这些对象是由交换链扩展自动创建的。这次我们必须自己创建一个。创建图像并将其填充数据类似于创建顶点缓冲区。我们将从创建临时资源并将其填充像素数据开始，然后将其复制到将用于渲染的最终图像对象。尽管可以为此创建临时映像，但Vulkan还允许您将像素从复制VkBuffer到映像，并且在某些硬件上，用于此目的的API实际上更快。我们将首先创建此缓冲区并用像素值填充它，然后将创建图像以将像素复制到该缓冲区。创建映像与创建缓冲区没有太大区别。正如我们之前所见，它涉及到查询内存需求，分配设备内存并对其进行绑定。
/// 但是，在处理图像时，我们需要注意一些其他事项。图像的布局可能会影响像素在内存中的组织方式。由于图形硬件的工作方式，例如，仅逐行存储像素可能不会导致最佳性能。对图像执行任何操作时，必须确保它们具有最适合在该操作中使用的布局。当指定渲染通道时，我们实际上已经看到了其中一些布局：
/// IMAGE_LAYOUT_PRESENT_SRC_KHR：最适合展示
/// IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL：最适合作为从片段着色器写入颜色的附件
/// IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL：最适合作为转移操作中的来源，例如 vkCmdCopyImageToBuffer
/// IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL：最适合作为转移操作中的目的地，例如 vkCmdCopyBufferToImage
/// IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL：最适合从着色器采样
///
/// 转换图像布局的最常见方法之一是管道屏障。流水线屏障主要用于同步对资源的访问，例如确保在读取图像之前已将其写入，但是它们也可以用于转换布局。
///
pub fn main() {
    // 构建顶点数据
    generator_vertices();

    let events = EventLoop::new();
    let mut hello = HelloTriangleApplication::default();
    let win = hello.run(&events);
    hello.main_loop(events, win);
}

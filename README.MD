# vulkan 教程

[![LICENSE](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

A Rust implementation of the [Vulkan Tutorial](https://vulkan-tutorial.com) based on [ash crate](https://crates.io/crates/ash).

## Status

The environment configuration is same as ash. See [here](https://github.com/MaikKlein/ash#example) for more detail.

Vulkan is quite hard. All examples on official website have been finished. :beers:

## Usage

Run any example with the following command:

```shell
$ cargo run --features=debug 1
```

## Snapshot

![rendering_and_presentation](snapshot/rendering_and_presentation.png)
![vertex_buffer_creation](snapshot/vertex_buffer_creation.png)
![index_buffer](snapshot/index_buffer.png)
![descriptor_pool_and_sets](snapshot/descriptor_pool_and_sets.png)
![combined_image_sampler](snapshot/combined_image_sampler.png)
![combined_image_sampler1](snapshot/combined_image_sampler1.png)
![depth_buffering](snapshot/depth_buffering.png)

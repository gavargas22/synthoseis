//! wgpu adapter probe + WGSL tile fuse dispatch.
//!
//! Compiled only with the `wgpu` feature. When no adapter is present (typical
//! CI `ubuntu-latest` without a GPU/ICD), callers fall back to the CPU path.

use std::sync::{Mutex, OnceLock};
use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;
use crate::FuseBackend;

const SHADER: &str = include_str!("shaders/fuse_tile.wgsl");

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct FuseParams {
    nj: u32, nk: u32, i0: u32, i1: u32, j0: u32, j1: u32,
    wavelet_len: u32, wavelet_off: u32, angle_deg: f32, mode: u32, sin_t: f32, cos_t: f32,
}

struct GpuContext {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    bind_layout: wgpu::BindGroupLayout,
    adapter_name: String,
}

static GPU: OnceLock<Option<Mutex<GpuContext>>> = OnceLock::new();

fn storage_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn init_gpu() -> Option<Mutex<GpuContext>> {
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
        backends: wgpu::Backends::VULKAN | wgpu::Backends::METAL | wgpu::Backends::DX12 | wgpu::Backends::GL,
        ..Default::default()
    });
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
    })).or_else(|| pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::LowPower,
        compatible_surface: None,
        force_fallback_adapter: true,
    })))?;
    let info = adapter.get_info();
    let adapter_name = format!("{} ({:?}/{:?})", info.name, info.backend, info.device_type);
    let mut limits = adapter.limits();
    if limits.max_storage_buffers_per_shader_stage < 4 {
        limits.max_storage_buffers_per_shader_stage = 4;
    }
    let (device, queue) = pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            label: Some("synthoseis-gpu-fuse"),
            required_features: wgpu::Features::empty(),
            required_limits: limits,
            memory_hints: wgpu::MemoryHints::default(),
        },
        None,
    )).ok()?;
    device.on_uncaptured_error(Box::new(|err| {
        eprintln!("synthoseis-gpu wgpu error: {err}");
    }));
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("fuse_tile.wgsl"),
        source: wgpu::ShaderSource::Wgsl(SHADER.into()),
    });
    let bind_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("fuse_bind_layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            storage_entry(1, true),
            storage_entry(2, true),
            storage_entry(3, false),
            storage_entry(4, false),
        ],
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("fuse_pipeline_layout"),
        bind_group_layouts: &[&bind_layout],
        push_constant_ranges: &[],
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("fuse_pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });
    Some(Mutex::new(GpuContext { device, queue, pipeline, bind_layout, adapter_name }))
}

fn ctx() -> Option<&'static Mutex<GpuContext>> {
    GPU.get_or_init(init_gpu).as_ref()
}

pub fn gpu_device_available() -> bool { ctx().is_some() }

pub fn backend_status() -> String {
    match ctx() {
        Some(m) => match m.lock() {
            Ok(g) => format!("wgpu/WGSL ({})", g.adapter_name),
            Err(_) => "wgpu/WGSL (lock poisoned)".into(),
        },
        None => "cpu-software (no wgpu adapter)".into(),
    }
}

pub fn fuse_tile_wgpu(
    labels: &[u8],
    shape: [usize; 3],
    i0: usize,
    i1: usize,
    j0: usize,
    j1: usize,
    trends: &[Vec<f64>; 9],
    wavelet: &[f64],
    angle_deg: f64,
    tile_out: &mut [f32],
) -> Option<FuseBackend> {
    let gpu = ctx()?;
    let g = gpu.lock().ok()?;
    let [_ni, nj, nk] = shape;
    let ti = i1 - i0;
    let tj = j1 - j0;
    assert_eq!(tile_out.len(), ti * tj * nk);
    assert!(i1 <= shape[0] && j1 <= shape[1] && nk >= 1);
    for t in trends { assert_eq!(t.len(), nk); }
    assert_eq!(labels.len(), shape[0] * shape[1] * shape[2]);

    let labels_u32: Vec<u32> = labels.iter().map(|&b| b as u32).collect();
    let mut trends_wavelet = Vec::with_capacity(9 * nk + wavelet.len().max(1));
    for t in trends {
        for &v in t { trends_wavelet.push(v as f32); }
    }
    let wavelet_off = push_wavelet(&mut trends_wavelet, wavelet);

    let params = FuseParams {
        nj: nj as u32, nk: nk as u32, i0: i0 as u32, i1: i1 as u32,
        j0: j0 as u32, j1: j1 as u32,
        wavelet_len: wavelet.len() as u32, wavelet_off,
        angle_deg: angle_deg as f32, mode: 0,
        sin_t: angle_deg.to_radians().sin() as f32, cos_t: angle_deg.to_radians().cos() as f32,
    };
    run_fuse(&g, &params, &labels_u32, &trends_wavelet, ti * tj, tile_out)
}

fn push_wavelet(buf: &mut Vec<f32>, wavelet: &[f64]) -> u32 {
    let off = buf.len() as u32;
    if wavelet.is_empty() { buf.push(0.0); }
    else { for &v in wavelet { buf.push(v as f32); } }
    off
}

/// Property-mode dispatch: `vp`, `vs`, `rho` are `(n_traces, nk)` tile buffers.
pub fn fuse_props_tile_wgpu(
    vp: &[f32],
    vs: &[f32],
    rho: &[f32],
    nk: usize,
    wavelet: &[f64],
    angle_deg: f64,
    tile_out: &mut [f32],
) -> Option<FuseBackend> {
    let gpu = ctx()?;
    let g = gpu.lock().ok()?;
    let n = tile_out.len();
    assert!(nk >= 1 && n % nk == 0);
    assert!(vp.len() == n && vs.len() == n && rho.len() == n);
    let n_traces = n / nk;
    let mut props_wavelet = Vec::with_capacity(3 * n + wavelet.len().max(1));
    props_wavelet.extend_from_slice(vp);
    props_wavelet.extend_from_slice(vs);
    props_wavelet.extend_from_slice(rho);
    let wavelet_off = push_wavelet(&mut props_wavelet, wavelet);
    let params = FuseParams {
        nj: 1, nk: nk as u32, i0: 0, i1: n_traces as u32, j0: 0, j1: 1,
        wavelet_len: wavelet.len() as u32, wavelet_off,
        angle_deg: angle_deg as f32, mode: 1,
        sin_t: angle_deg.to_radians().sin() as f32, cos_t: angle_deg.to_radians().cos() as f32,
    };
    run_fuse(&g, &params, &[0u32], &props_wavelet, n_traces, tile_out)
}

fn run_fuse(
    g: &GpuContext,
    params: &FuseParams,
    labels_u32: &[u32],
    trends_wavelet: &[f32],
    n_traces: usize,
    tile_out: &mut [f32],
) -> Option<FuseBackend> {
    let device = &g.device;
    let queue = &g.queue;
    let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("fuse_params"), contents: bytemuck::bytes_of(params),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let labels_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("fuse_labels"), contents: bytemuck::cast_slice(labels_u32),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let tw_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("fuse_trends_wavelet"), contents: bytemuck::cast_slice(trends_wavelet),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let tile_bytes = (tile_out.len() * 4) as u64;
    let scratch_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("fuse_scratch"), size: tile_bytes.max(4),
        usage: wgpu::BufferUsages::STORAGE, mapped_at_creation: false,
    });
    let out_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("fuse_out"), size: tile_bytes.max(4),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false,
    });
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("fuse_staging"), size: tile_bytes.max(4),
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false,
    });
    let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("fuse_bind"), layout: &g.bind_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: params_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: labels_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: tw_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: scratch_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: out_buf.as_entire_binding() },
        ],
    });
    let wg = ((n_traces as u32) + 63) / 64;
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("fuse_encoder"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("fuse_pass"), timestamp_writes: None,
        });
        pass.set_pipeline(&g.pipeline);
        pass.set_bind_group(0, &bind, &[]);
        pass.dispatch_workgroups(wg.max(1), 1, 1);
    }
    encoder.copy_buffer_to_buffer(&out_buf, 0, &staging, 0, tile_bytes.max(4));
    queue.submit(Some(encoder.finish()));
    let slice = staging.slice(..tile_bytes.max(4));
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| { let _ = tx.send(r); });
    device.poll(wgpu::Maintain::Wait);
    rx.recv().ok()?.ok()?;
    let data = slice.get_mapped_range();
    let floats: &[f32] = bytemuck::cast_slice(&data);
    tile_out.copy_from_slice(&floats[..tile_out.len()]);
    drop(data);
    staging.unmap();
    Some(FuseBackend::Gpu)
}

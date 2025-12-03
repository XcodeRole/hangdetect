use crate::cuda_funcs;
use libc::c_uint;
use crate::monitor::{LaunchCUDAKernel, monitor_launch_cuda_kernel};
use std::ffi::{c_int, c_void};

pub extern "C" fn cudaLaunchKernel(
    func: *const c_void,
    grid_dim: cuda_funcs::Dim3,
    block_dim: cuda_funcs::Dim3,
    args: *const *const c_void,
    shared_mem: usize,
    stream: *mut c_void,
) -> c_int {
    monitor_launch_cuda_kernel(LaunchCUDAKernel::Runtime { func, stream }, || {
        cuda_funcs::launch_cuda_kernel(func, grid_dim, block_dim, args, shared_mem, stream)
    })
}

pub extern "C" fn cudaLaunchKernelExC(
    config: &cuda_funcs::CudaLaunchConfig,
    func: *const c_void,
    args: *mut *const c_void,
) -> c_int {
    monitor_launch_cuda_kernel(
        LaunchCUDAKernel::Runtime {
            func,
            stream: config.stream,
        },
        || cuda_funcs::launch_cuda_kernel_ex_c(config, func, args),
    )
}

pub extern "C" fn cuLaunchKernel(
    func: *const c_void,
    grid_dim_x: c_uint,
    grid_dim_y: c_uint,
    grid_dim_z: c_uint,
    block_dim_x: c_uint,
    block_dim_y: c_uint,
    block_dim_z: c_uint,
    shared_mem: c_uint,
    stream: *const c_void,
    kernel_params: *mut *const c_void,
    extra: *mut *const c_void,
) -> c_int {
    monitor_launch_cuda_kernel(LaunchCUDAKernel::Driver { func, stream }, || {
        cuda_funcs::launch_cu_kernel(
            func,
            grid_dim_x,
            grid_dim_y,
            grid_dim_z,
            block_dim_x,
            block_dim_y,
            block_dim_z,
            shared_mem,
            stream,
            kernel_params,
            extra,
        )
    })
}

pub extern "C" fn cuLaunchKernelEx(
    config: &cuda_funcs::CuLaunchConfig,
    func: *const c_void,
    kernel_params: *mut *const c_void,
    extra: *mut *const c_void,
) -> c_int {
    monitor_launch_cuda_kernel(
        LaunchCUDAKernel::Driver {
            func,
            stream: config.stream,
        },
        || cuda_funcs::launch_cu_kernel_ex(config, func, kernel_params, extra),
    )
}
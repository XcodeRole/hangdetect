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

// NCCL Wrappers

#[unsafe(no_mangle)]
pub extern "C" fn ncclAllReduce(
    sendbuff: *const c_void,
    recvbuff: *mut c_void,
    count: usize,
    datatype: cuda_funcs::ncclDataType_t,
    op: cuda_funcs::ncclRedOp_t,
    comm: cuda_funcs::ncclComm_t,
    stream: *mut c_void,
) -> cuda_funcs::ncclResult_t {
    monitor_launch_cuda_kernel(
        LaunchCUDAKernel::Nccl {
            name: "ncclAllReduce",
            stream,
        },
        || {
            match cuda_funcs::nccl_all_reduce(sendbuff, recvbuff, count, datatype, op, comm, stream)
            {
                0 => Ok(()),
                e => Err(cuda_funcs::CUDAError { code: e }),
            }
        },
    )
}

#[unsafe(no_mangle)]
pub extern "C" fn ncclBroadcast(
    sendbuff: *const c_void,
    recvbuff: *mut c_void,
    count: usize,
    datatype: cuda_funcs::ncclDataType_t,
    root: c_int,
    comm: cuda_funcs::ncclComm_t,
    stream: *mut c_void,
) -> cuda_funcs::ncclResult_t {
    monitor_launch_cuda_kernel(
        LaunchCUDAKernel::Nccl {
            name: "ncclBroadcast",
            stream,
        },
        || {
            match cuda_funcs::nccl_broadcast(sendbuff, recvbuff, count, datatype, root, comm, stream)
            {
                0 => Ok(()),
                e => Err(cuda_funcs::CUDAError { code: e }),
            }
        },
    )
}

#[unsafe(no_mangle)]
pub extern "C" fn ncclBcast(
    buff: *mut c_void,
    count: usize,
    datatype: cuda_funcs::ncclDataType_t,
    root: c_int,
    comm: cuda_funcs::ncclComm_t,
    stream: *mut c_void,
) -> cuda_funcs::ncclResult_t {
    monitor_launch_cuda_kernel(
        LaunchCUDAKernel::Nccl {
            name: "ncclBcast",
            stream,
        },
        || {
            match cuda_funcs::nccl_bcast(buff, count, datatype, root, comm, stream) {
                0 => Ok(()),
                e => Err(cuda_funcs::CUDAError { code: e }),
            }
        },
    )
}

#[unsafe(no_mangle)]
pub extern "C" fn ncclReduce(
    sendbuff: *const c_void,
    recvbuff: *mut c_void,
    count: usize,
    datatype: cuda_funcs::ncclDataType_t,
    op: cuda_funcs::ncclRedOp_t,
    root: c_int,
    comm: cuda_funcs::ncclComm_t,
    stream: *mut c_void,
) -> cuda_funcs::ncclResult_t {
    monitor_launch_cuda_kernel(
        LaunchCUDAKernel::Nccl {
            name: "ncclReduce",
            stream,
        },
        || {
            match cuda_funcs::nccl_reduce(sendbuff, recvbuff, count, datatype, op, root, comm, stream)
            {
                0 => Ok(()),
                e => Err(cuda_funcs::CUDAError { code: e }),
            }
        },
    )
}

#[unsafe(no_mangle)]
pub extern "C" fn ncclAllGather(
    sendbuff: *const c_void,
    recvbuff: *mut c_void,
    sendcount: usize,
    datatype: cuda_funcs::ncclDataType_t,
    comm: cuda_funcs::ncclComm_t,
    stream: *mut c_void,
) -> cuda_funcs::ncclResult_t {
    monitor_launch_cuda_kernel(
        LaunchCUDAKernel::Nccl {
            name: "ncclAllGather",
            stream,
        },
        || {
            match cuda_funcs::nccl_all_gather(sendbuff, recvbuff, sendcount, datatype, comm, stream)
            {
                0 => Ok(()),
                e => Err(cuda_funcs::CUDAError { code: e }),
            }
        },
    )
}

#[unsafe(no_mangle)]
pub extern "C" fn ncclReduceScatter(
    sendbuff: *const c_void,
    recvbuff: *mut c_void,
    recvcount: usize,
    datatype: cuda_funcs::ncclDataType_t,
    op: cuda_funcs::ncclRedOp_t,
    comm: cuda_funcs::ncclComm_t,
    stream: *mut c_void,
) -> cuda_funcs::ncclResult_t {
    monitor_launch_cuda_kernel(
        LaunchCUDAKernel::Nccl {
            name: "ncclReduceScatter",
            stream,
        },
        || {
            match cuda_funcs::nccl_reduce_scatter(
                sendbuff, recvbuff, recvcount, datatype, op, comm, stream,
            ) {
                0 => Ok(()),
                e => Err(cuda_funcs::CUDAError { code: e }),
            }
        },
    )
}

#[unsafe(no_mangle)]
pub extern "C" fn ncclAlltoAll(
    sendbuff: *const c_void,
    recvbuff: *mut c_void,
    count: usize,
    datatype: cuda_funcs::ncclDataType_t,
    comm: cuda_funcs::ncclComm_t,
    stream: *mut c_void,
) -> cuda_funcs::ncclResult_t {
    monitor_launch_cuda_kernel(
        LaunchCUDAKernel::Nccl {
            name: "ncclAlltoAll",
            stream,
        },
        || {
            match cuda_funcs::nccl_all_to_all(sendbuff, recvbuff, count, datatype, comm, stream) {
                0 => Ok(()),
                e => Err(cuda_funcs::CUDAError { code: e }),
            }
        },
    )
}

#[unsafe(no_mangle)]
pub extern "C" fn ncclGather(
    sendbuff: *const c_void,
    recvbuff: *mut c_void,
    count: usize,
    datatype: cuda_funcs::ncclDataType_t,
    root: c_int,
    comm: cuda_funcs::ncclComm_t,
    stream: *mut c_void,
) -> cuda_funcs::ncclResult_t {
    monitor_launch_cuda_kernel(
        LaunchCUDAKernel::Nccl {
            name: "ncclGather",
            stream,
        },
        || {
            match cuda_funcs::nccl_gather(sendbuff, recvbuff, count, datatype, root, comm, stream) {
                0 => Ok(()),
                e => Err(cuda_funcs::CUDAError { code: e }),
            }
        },
    )
}

#[unsafe(no_mangle)]
pub extern "C" fn ncclScatter(
    sendbuff: *const c_void,
    recvbuff: *mut c_void,
    count: usize,
    datatype: cuda_funcs::ncclDataType_t,
    root: c_int,
    comm: cuda_funcs::ncclComm_t,
    stream: *mut c_void,
) -> cuda_funcs::ncclResult_t {
    monitor_launch_cuda_kernel(
        LaunchCUDAKernel::Nccl {
            name: "ncclScatter",
            stream,
        },
        || {
            match cuda_funcs::nccl_scatter(sendbuff, recvbuff, count, datatype, root, comm, stream) {
                0 => Ok(()),
                e => Err(cuda_funcs::CUDAError { code: e }),
            }
        },
    )
}

#[unsafe(no_mangle)]
pub extern "C" fn ncclSend(
    sendbuff: *const c_void,
    count: usize,
    datatype: cuda_funcs::ncclDataType_t,
    peer: c_int,
    comm: cuda_funcs::ncclComm_t,
    stream: *mut c_void,
) -> cuda_funcs::ncclResult_t {
    monitor_launch_cuda_kernel(
        LaunchCUDAKernel::Nccl {
            name: "ncclSend",
            stream,
        },
        || {
            match cuda_funcs::nccl_send(sendbuff, count, datatype, peer, comm, stream) {
                0 => Ok(()),
                e => Err(cuda_funcs::CUDAError { code: e }),
            }
        },
    )
}

#[unsafe(no_mangle)]
pub extern "C" fn ncclRecv(
    recvbuff: *mut c_void,
    count: usize,
    datatype: cuda_funcs::ncclDataType_t,
    peer: c_int,
    comm: cuda_funcs::ncclComm_t,
    stream: *mut c_void,
) -> cuda_funcs::ncclResult_t {
    monitor_launch_cuda_kernel(
        LaunchCUDAKernel::Nccl {
            name: "ncclRecv",
            stream,
        },
        || {
            match cuda_funcs::nccl_recv(recvbuff, count, datatype, peer, comm, stream) {
                0 => Ok(()),
                e => Err(cuda_funcs::CUDAError { code: e }),
            }
        },
    )
}
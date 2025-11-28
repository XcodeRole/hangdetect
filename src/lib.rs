use crate::cuda_funcs::{CuLaunchConfig, CudaLaunchConfig};
use libc::{c_char, c_uint, c_ulonglong};
use monitor::{LaunchCUDAKernel, NCCLCommunication, monitor_launch_cuda_kernel, monitor_nccl_communication};
use std::ffi::{c_int, c_void};

mod cuda_funcs;
mod init;
mod logger;
mod nccl_funcs;

mod monitor;

// CUDA Runtime API hooks
#[unsafe(no_mangle)]
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

#[unsafe(no_mangle)]
pub extern "C" fn cudaLaunchKernelExC(
    config: &CudaLaunchConfig,
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

// CUDA Driver API hooks
#[unsafe(no_mangle)]
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

#[unsafe(no_mangle)]
pub extern "C" fn cuLaunchKernelEx(
    config: &CuLaunchConfig,
    func: *const c_void,
    args: *mut *const c_void,
) -> c_int {
    monitor_launch_cuda_kernel(
        LaunchCUDAKernel::Driver {
            func,
            stream: config.stream,
        },
        || cuda_funcs::launch_cu_kernel_ex(config, func, args),
    )
}

// NCCL Collective Communication API hooks
#[unsafe(no_mangle)]
pub extern "C" fn ncclAllReduce(
    sendbuff: *const c_void,
    recvbuff: *mut c_void,
    count: c_ulonglong,
    datatype: c_uint,
    op: c_uint,
    comm: *const c_void,
    stream: *const c_void,
) -> c_int {
    monitor_nccl_communication(NCCLCommunication::AllReduce {
        comm,
        stream,
        count,
        datatype,
        op,
    }, || {
        nccl_funcs::nccl_all_reduce(sendbuff, recvbuff, count, datatype, op, comm, stream)
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn ncclBroadcast(
    sendbuff: *const c_void,
    recvbuff: *mut c_void,
    count: c_ulonglong,
    datatype: c_uint,
    root: c_uint,
    comm: *const c_void,
    stream: *const c_void,
) -> c_int {
    monitor_nccl_communication(NCCLCommunication::Broadcast {
        comm,
        stream,
        count,
        datatype,
        root,
    }, || {
        nccl_funcs::nccl_broadcast(sendbuff, recvbuff, count, datatype, root, comm, stream)
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn ncclReduce(
    sendbuff: *const c_void,
    recvbuff: *mut c_void,
    count: c_ulonglong,
    datatype: c_uint,
    op: c_uint,
    root: c_uint,
    comm: *const c_void,
    stream: *const c_void,
) -> c_int {
    monitor_nccl_communication(NCCLCommunication::Reduce {
        comm,
        stream,
        count,
        datatype,
        op,
        root,
    }, || {
        nccl_funcs::nccl_reduce(sendbuff, recvbuff, count, datatype, op, root, comm, stream)
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn ncclAllGather(
    sendbuff: *const c_void,
    recvbuff: *mut c_void,
    sendcount: c_ulonglong,
    datatype: c_uint,
    comm: *const c_void,
    stream: *const c_void,
) -> c_int {
    monitor_nccl_communication(NCCLCommunication::AllGather {
        comm,
        stream,
        sendcount,
        datatype,
    }, || {
        nccl_funcs::nccl_all_gather(sendbuff, recvbuff, sendcount, datatype, comm, stream)
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn ncclReduceScatter(
    sendbuff: *const c_void,
    recvbuff: *mut c_void,
    recvcount: c_ulonglong,
    datatype: c_uint,
    op: c_uint,
    comm: *const c_void,
    stream: *const c_void,
) -> c_int {
    monitor_nccl_communication(NCCLCommunication::ReduceScatter {
        comm,
        stream,
        recvcount,
        datatype,
        op,
    }, || {
        nccl_funcs::nccl_reduce_scatter(sendbuff, recvbuff, recvcount, datatype, op, comm, stream)
    })
}

// Note: ncclAlltoAll, ncclGather, ncclScatter 在 NCCL 2.28 版本之后才有，在此之前应该是通过 send / recv 实现的

#[unsafe(no_mangle)]
pub extern "C" fn ncclAlltoAll(
    sendbuff: *const c_void,
    recvbuff: *mut c_void,
    count: c_ulonglong,
    datatype: c_uint,
    comm: *const c_void,
    stream: *const c_void,
) -> c_int {
    monitor_nccl_communication(NCCLCommunication::AlltoAll {
        comm,
        stream,
        count,
        datatype,
    }, || {
        nccl_funcs::nccl_alltoall(sendbuff, recvbuff, count, datatype, comm, stream)
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn ncclGather(
    sendbuff: *const c_void,
    recvbuff: *mut c_void,
    sendcount: c_ulonglong,
    datatype: c_uint,
    root: c_uint,
    comm: *const c_void,
    stream: *const c_void,
) -> c_int {
    monitor_nccl_communication(NCCLCommunication::Gather {
        comm,
        stream,
        sendcount,
        datatype,
        root,
    }, || {
        nccl_funcs::nccl_gather(sendbuff, recvbuff, sendcount, datatype, root, comm, stream)
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn ncclScatter(
    sendbuff: *const c_void,
    recvbuff: *mut c_void,
    recvcount: c_ulonglong,
    datatype: c_uint,
    root: c_uint,
    comm: *const c_void,
    stream: *const c_void,
) -> c_int {
    monitor_nccl_communication(NCCLCommunication::Scatter {
        comm,
        stream,
        recvcount,
        datatype,
        root,
    }, || {
        nccl_funcs::nccl_scatter(sendbuff, recvbuff, recvcount, datatype, root, comm, stream)
    })
}

// NCCL Point-to-Point Communication API hooks
#[unsafe(no_mangle)]
pub extern "C" fn ncclSend(
    sendbuff: *const c_void,
    count: c_ulonglong,
    datatype: c_uint,
    peer: c_uint,
    comm: *const c_void,
    stream: *const c_void,
) -> c_int {
    monitor_nccl_communication(NCCLCommunication::Send {
        comm,
        stream,
        count,
        datatype,
        peer,
    }, || {
        nccl_funcs::nccl_send(sendbuff, count, datatype, peer, comm, stream)
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn ncclRecv(
    recvbuff: *mut c_void,
    count: c_ulonglong,
    datatype: c_uint,
    peer: c_uint,
    comm: *const c_void,
    stream: *const c_void,
) -> c_int {
    monitor_nccl_communication(NCCLCommunication::Recv {
        comm,
        stream,
        count,
        datatype,
        peer,
    }, || {
        nccl_funcs::nccl_recv(recvbuff, count, datatype, peer, comm, stream)
    })
}

// Settings APIs
#[unsafe(no_mangle)]
pub extern "C" fn hangdetect_set_enable(enabled: bool) {
    monitor::set_hang_detection_enabled(enabled);
}

#[unsafe(no_mangle)]
pub extern "C" fn hangdetect_set_kernel_exec_label(label: *const c_char) {
    if label.is_null() {
        monitor::set_kernel_exec_time_user_label("");
    } else {
        let c_str = unsafe { std::ffi::CStr::from_ptr(label) };
        if let Ok(str_slice) = c_str.to_str() {
            monitor::set_kernel_exec_time_user_label(str_slice);
        } else {
            log::warn!("hangdetect_set_kernel_exec_label: invalid UTF-8 string");
        }
    }
}

mod aspects;
mod error;
mod filter;
mod kernel_exec_time_aspect;
mod launch_cuda_kernel;
mod launch_nccl_comm;
mod logging_aspect;
mod monitor_aspect;
mod thread_local_enabler;

use crate::cuda_funcs;
use crate::nccl_funcs::NCCLError;
use cuda_funcs::CUDAError;
pub use launch_cuda_kernel::LaunchCUDAKernel;
pub use launch_nccl_comm::NCCLCommunication;
use libc::c_int;

use aspects::ASPECTS;
pub use kernel_exec_time_aspect::set_kernel_exec_time_user_label;
pub use monitor_aspect::Operation;
pub use thread_local_enabler::set_hang_detection_enabled;

pub fn monitor_launch_cuda_kernel<F>(launch: LaunchCUDAKernel, f: F) -> c_int
where
    F: FnOnce() -> Result<(), CUDAError>,
{
    let op = Operation::LaunchCUDAKernel(&launch);
    match ASPECTS.before_call(&op) {
        Err(err) => match err {
            error::MonitorError::CUDAError(cuda_err) => return cuda_err.code,
            error::MonitorError::Internal(err) => {
                panic!("monitor before call internal error: {}", err);
            }
        },
        Ok(()) => {}
    }
    let retv = match f() {
        Err(err) => err.code,
        Ok(()) => 0,
    };

    let op = Operation::LaunchCUDAKernel(&launch);
    match ASPECTS.after_call(&op) {
        Err(err) => match err {
            error::MonitorError::CUDAError(cuda_err) => return cuda_err.code,
            error::MonitorError::Internal(err) => {
                panic!("monitor after call internal error: {}", err);
            }
        },
        Ok(()) => {}
    }
    retv
}

pub fn monitor_nccl_communication<F>(comm: NCCLCommunication, f: F) -> c_int
where
    F: FnOnce() -> Result<(), NCCLError>,
{
    let op = Operation::NCCLCommunication(&comm);
    match ASPECTS.before_call(&op) {
        Err(err) => match err {
            error::MonitorError::CUDAError(cuda_err) => return cuda_err.code,
            error::MonitorError::Internal(err) => {
                panic!("monitor before call internal error: {}", err);
            }
        },
        Ok(()) => {}
    }
    let retv = match f() {
        Err(err) => err.code,
        Ok(()) => 0,
    };

    let op = Operation::NCCLCommunication(&comm);
    match ASPECTS.after_call(&op) {
        Err(err) => match err {
            error::MonitorError::CUDAError(cuda_err) => return cuda_err.code,
            error::MonitorError::Internal(err) => {
                panic!("monitor after call internal error: {}", err);
            }
        },
        Ok(()) => {}
    }
    retv
}

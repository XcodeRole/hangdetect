mod aspects;
mod error;
mod filter;
mod kernel_exec_time_aspect;
mod launch_cuda_kernel;
mod logging_aspect;
mod monitor_aspect;
mod thread_local_enabler;
mod kernel_name_filter;

use crate::cuda_funcs;
use cuda_funcs::CUDAError;
pub use launch_cuda_kernel::LaunchCUDAKernel;
use libc::c_int;

use aspects::ASPECTS;
pub use kernel_exec_time_aspect::set_kernel_exec_time_user_label;
pub use thread_local_enabler::set_hang_detection_enabled;

pub fn monitor_launch_cuda_kernel<F>(launch: LaunchCUDAKernel, f: F) -> c_int
where
    F: FnOnce() -> Result<(), CUDAError>,
{
    match ASPECTS.before_call(&launch) {
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

    match ASPECTS.after_call(&launch) {
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

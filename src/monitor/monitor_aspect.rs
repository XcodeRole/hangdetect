use crate::monitor::LaunchCUDAKernel;
use crate::monitor::NCCLCommunication;
use crate::monitor::error::MonitorError;
use std::fmt::{Display, Formatter};

pub enum Operation<'a> {
    LaunchCUDAKernel(&'a LaunchCUDAKernel),
    NCCLCommunication(&'a NCCLCommunication),
}

impl<'a> Display for Operation<'a> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Operation::LaunchCUDAKernel(launch) => write!(f, "{}", launch),
            Operation::NCCLCommunication(comm) => write!(f, "{}", comm),
        }
    }
}

impl <'a> Operation<'a> {
    pub fn name(&self) -> Result<String, MonitorError> {
        match self {
            Operation::LaunchCUDAKernel(launch) => {
                let func_name = launch.func_name()?;
                Ok(func_name.display_name().to_string())
            }
            Operation::NCCLCommunication(comm) => Ok(comm.api_name().to_string()),
        }
    }

    pub fn stream(&self) -> *const std::ffi::c_void {
        match self {
            Operation::LaunchCUDAKernel(launch) => launch.stream(),
            Operation::NCCLCommunication(comm) => comm.stream(),
        }
    }
}

pub trait MonitorAspect: Send + Sync {
    fn before_call(&self, op: &Operation<'_>) -> Result<(), MonitorError>;

    fn after_call(&self, op: &Operation<'_>) -> Result<(), MonitorError>;
}

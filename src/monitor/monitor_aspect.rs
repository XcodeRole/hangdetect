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

pub trait MonitorAspect: Send + Sync {
    fn before_call(&self, op: &Operation<'_>) -> Result<(), MonitorError>;

    fn after_call(&self, op: &Operation<'_>) -> Result<(), MonitorError>;
}

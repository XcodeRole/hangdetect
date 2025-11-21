use crate::monitor::LaunchCUDAKernel;
use crate::monitor::NCCLCommunication;
use crate::monitor::error::MonitorError;

pub trait MonitorAspect: Send + Sync {
    fn before_call(&self, launch: &LaunchCUDAKernel) -> Result<(), MonitorError>;

    fn after_call(&self, launch: &LaunchCUDAKernel) -> Result<(), MonitorError>;

    fn before_nccl_call(&self, comm: &NCCLCommunication) -> Result<(), MonitorError>;

    fn after_nccl_call(&self, comm: &NCCLCommunication) -> Result<(), MonitorError>;
}

use super::monitor_aspect::{MonitorAspect, Operation};

pub struct LoggingAspect {}

impl MonitorAspect for LoggingAspect {
    fn before_call(&self, op: &Operation<'_>) -> Result<(), crate::monitor::error::MonitorError> {
        match op {
            Operation::LaunchCUDAKernel(launch) => {
                log::info!("Launching CUDA kernel: {}", launch);
            }
            Operation::NCCLCommunication(comm) => {
                log::info!("Starting NCCL communication: {}", comm);
            }
        }
        Ok(())
    }

    fn after_call(&self, _op: &Operation<'_>) -> Result<(), crate::monitor::error::MonitorError> {
        Ok(())
    }
}

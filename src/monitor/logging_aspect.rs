use super::monitor_aspect::MonitorAspect;

pub struct LoggingAspect {}

impl MonitorAspect for LoggingAspect {
    fn before_call(
        &self,
        launch: &crate::monitor::LaunchCUDAKernel,
    ) -> Result<(), crate::monitor::error::MonitorError> {
        log::info!("Launching CUDA kernel: {}", launch);
        Ok(())
    }

    fn after_call(
        &self,
        _launch: &crate::monitor::LaunchCUDAKernel,
    ) -> Result<(), crate::monitor::error::MonitorError> {
        Ok(())
    }

    fn before_nccl_call(
        &self,
        comm: &crate::monitor::NCCLCommunication,
    ) -> Result<(), crate::monitor::error::MonitorError> {
        log::info!("Starting NCCL communication: {}", comm);
        Ok(())
    }

    fn after_nccl_call(
        &self,
        _comm: &crate::monitor::NCCLCommunication,
    ) -> Result<(), crate::monitor::error::MonitorError> {
        Ok(())
    }
}

use super::monitor_aspect::{MonitorAspect, Operation};

pub struct LoggingAspect {}

impl MonitorAspect for LoggingAspect {
    fn before_call(&self, op: &Operation<'_>) -> Result<(), crate::monitor::error::MonitorError> {
        log::info!("Launching {}", op);
        Ok(())
    }

    fn after_call(&self, _op: &Operation<'_>) -> Result<(), crate::monitor::error::MonitorError> {
        Ok(())
    }
}

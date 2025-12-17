use super::filter::Filter;
use crate::monitor::LaunchCUDAKernel;
use std::sync::atomic::{AtomicBool, Ordering};
static HANG_DETECTION_ENABLED: AtomicBool = AtomicBool::new(false);

pub struct Enabler {}

impl Filter for Enabler {
    fn filter(&self, _launch: &LaunchCUDAKernel) -> bool {
        HANG_DETECTION_ENABLED.load(Ordering::Relaxed)
    }
}

pub fn set_hang_detection_enabled(enabled: bool) {
    HANG_DETECTION_ENABLED.store(enabled, Ordering::Relaxed);
    log::info!("set HANG_DETECTION_ENABLED to {}", enabled);
}

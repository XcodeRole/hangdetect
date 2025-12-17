use super::filter::Filter;
use crate::monitor::LaunchCUDAKernel;
use once_cell::sync::Lazy;
use std::sync::atomic::{AtomicBool, Ordering};

static HANG_DETECTION_ENABLED: Lazy<AtomicBool> = Lazy::new(|| {
    let enabled = std::env::var("HANG_DETECTION_ENABLED")
        .map(|v| v == "1")
        .unwrap_or(false);
    log::info!("HANG_DETECTION_ENABLED [{}]", enabled);
    AtomicBool::new(enabled)
});

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

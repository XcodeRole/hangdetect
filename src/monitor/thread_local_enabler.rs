use super::filter::Filter;
use crate::monitor::LaunchCUDAKernel;
use std::cell::RefCell;
thread_local! {
    static HANG_DETECTION_ENABLED: RefCell<Option<bool>> = RefCell::new(None);
}

pub struct ThreadLocalEnabler {}

impl Filter for ThreadLocalEnabler {
    fn filter(&self, _launch: &LaunchCUDAKernel) -> bool {
        HANG_DETECTION_ENABLED.with(|h| {
            let mut flag = h.borrow_mut();

            if flag.is_none() {
                let enabled = std::env::var("HANG_DETECTION_ENABLED")
                    .map(|v| v == "1")
                    .unwrap_or(false);
                flag.replace(enabled);
                log::info!("HANG_DETECTION_ENABLED [{}]", enabled);
            }

            let enabled = flag.unwrap();
            enabled
        })
    }
}

pub fn set_hang_detection_enabled(enabled: bool) {
    HANG_DETECTION_ENABLED.with(|h| {
        h.borrow_mut().replace(enabled);
    });
}

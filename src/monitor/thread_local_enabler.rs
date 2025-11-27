use super::filter::Filter;
use super::monitor_aspect::Operation;
use std::cell::RefCell;
thread_local! {
    static HANG_DETECTION_ENABLED: RefCell<Option<bool>> = RefCell::new(None);
}

pub struct ThreadLocalEnabler {}

impl ThreadLocalEnabler {
    fn is_enabled(&self) -> bool {
        _ = std::env::vars();
        HANG_DETECTION_ENABLED.with(|h| {
            let mut flag = h.borrow_mut();

            if flag.is_none() {
                let enabled = option_env!("HANG_DETECTION_ENABLED").unwrap_or_else(|| "0") == "1";
                flag.replace(enabled);
                log::info!("HANG_DETECTION_ENABLED [{}]", enabled);
            }

            flag.unwrap()
        })
    }
}

impl Filter for ThreadLocalEnabler {
    fn filter(&self, _op: &Operation<'_>) -> bool {
        self.is_enabled()
    }
}
pub fn set_hang_detection_enabled(enabled: bool) {
    HANG_DETECTION_ENABLED.with(|h| {
        let mut flag = h.borrow_mut();
        flag.replace(enabled);
    });
}

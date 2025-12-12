use core::ffi::c_char;

use crate::monitor;

// Settings APIs exposed via C ABI for Python/FFI callers.
// These are thread-local switches and labels that control hangdetect behavior.

#[unsafe(no_mangle)]
pub extern "C" fn hangdetect_set_enable(enabled: bool) {
    monitor::set_hang_detection_enabled(enabled);
}

#[unsafe(no_mangle)]
pub extern "C" fn hangdetect_set_kernel_exec_label(label: *const c_char) {
    if label.is_null() {
        monitor::set_kernel_exec_time_user_label("");
        return;
    }

    unsafe {
        let c_str = core::ffi::CStr::from_ptr(label);
        if let Ok(str_slice) = c_str.to_str() {
            monitor::set_kernel_exec_time_user_label(str_slice);
        } else {
            log::warn!("hangdetect_set_kernel_exec_label: invalid UTF-8 string");
        }
    }
}

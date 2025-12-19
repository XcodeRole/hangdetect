use core::ffi::c_char;

use crate::monitor;
use libc::{ftruncate, mmap, off_t, shm_open, MAP_FAILED, MAP_SHARED, O_CREAT, O_RDWR, PROT_READ, PROT_WRITE};
use std::ffi::CString;

#[repr(C)]
struct FunctionPointerBlock {
    enable_fn: u64,
    label_fn: u64,
}

// Settings APIs exposed via C ABI for FFI callers.
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
            eprintln!("[hangdetect][audit] hangdetect_set_kernel_exec_label: invalid UTF-8 string");
        }
    }
}

pub fn export_ffi_control_functions() {
    // Export control function pointers via shared memory named by PID.
    // supports multi-process (e.g. torchrun) by giving each process its own block.
    unsafe {
        let enable_fn: extern "C" fn(bool) = hangdetect_set_enable;
        let enable_addr = enable_fn as usize;

        let label_fn: extern "C" fn(*const c_char) =
            hangdetect_set_kernel_exec_label;
        let label_addr = label_fn as usize;

        let pid = std::process::id();
        let shm_name = format!("/hangdetect_ctl_{}", pid);
        let c_name = CString::new(shm_name).unwrap();

        // Create/Open shared memory for this process
        let fd = shm_open(c_name.as_ptr(), O_CREAT | O_RDWR, 0o666);
        if fd >= 0 {
            let size = std::mem::size_of::<FunctionPointerBlock>();
            if ftruncate(fd, size as off_t) == 0 {
                let ptr = mmap(
                    std::ptr::null_mut(),
                    size,
                    PROT_READ | PROT_WRITE,
                    MAP_SHARED,
                    fd,
                    0,
                );
                if ptr != MAP_FAILED {
                    let block = ptr as *mut FunctionPointerBlock;
                    (*block).enable_fn = enable_addr as u64;
                    (*block).label_fn = label_addr as u64;
                } else {
                    eprintln!("[hangdetect][audit] mmap failed");
                }
            } else {
                eprintln!("[hangdetect][audit] ftruncate failed");
            }
        } else {
            eprintln!("[hangdetect][audit] shm_open failed");
        }
    }
}

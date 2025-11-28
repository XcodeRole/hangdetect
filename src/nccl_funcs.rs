use crate::init::init;
use libc::{c_int, c_uint, c_ulonglong};
use log::{debug, error, warn};
use std::ffi::c_void;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::sync::Once;
use once_cell::sync::OnceCell;
use std::ffi::CString;

// NCCL Collective Communication Functions
type NcclAllReduce = unsafe extern "C" fn(
    sendbuff: *const c_void,
    recvbuff: *mut c_void,
    count: c_ulonglong,
    datatype: c_uint,
    op: c_uint,
    comm: *const c_void,
    stream: *const c_void,
) -> c_int;

type NcclBroadcast = unsafe extern "C" fn(
    sendbuff: *const c_void,
    recvbuff: *mut c_void,
    count: c_ulonglong,
    datatype: c_uint,
    root: c_uint,
    comm: *const c_void,
    stream: *const c_void,
) -> c_int;

type NcclReduce = unsafe extern "C" fn(
    sendbuff: *const c_void,
    recvbuff: *mut c_void,
    count: c_ulonglong,
    datatype: c_uint,
    op: c_uint,
    root: c_uint,
    comm: *const c_void,
    stream: *const c_void,
) -> c_int;

type NcclAllGather = unsafe extern "C" fn(
    sendbuff: *const c_void,
    recvbuff: *mut c_void,
    sendcount: c_ulonglong,
    datatype: c_uint,
    comm: *const c_void,
    stream: *const c_void,
) -> c_int;

type NcclReduceScatter = unsafe extern "C" fn(
    sendbuff: *const c_void,
    recvbuff: *mut c_void,
    recvcount: c_ulonglong,
    datatype: c_uint,
    op: c_uint,
    comm: *const c_void,
    stream: *const c_void,
) -> c_int;

type NcclAlltoAll = unsafe extern "C" fn(
    sendbuff: *const c_void,
    recvbuff: *mut c_void,
    count: c_ulonglong,
    datatype: c_uint,
    comm: *const c_void,
    stream: *const c_void,
) -> c_int;

type NcclGather = unsafe extern "C" fn(
    sendbuff: *const c_void,
    recvbuff: *mut c_void,
    sendcount: c_ulonglong,
    datatype: c_uint,
    root: c_uint,
    comm: *const c_void,
    stream: *const c_void,
) -> c_int;

type NcclScatter = unsafe extern "C" fn(
    sendbuff: *const c_void,
    recvbuff: *mut c_void,
    recvcount: c_ulonglong,
    datatype: c_uint,
    root: c_uint,
    comm: *const c_void,
    stream: *const c_void,
) -> c_int;

// NCCL Point-to-Point Communication Functions
type NcclSend = unsafe extern "C" fn(
    sendbuff: *const c_void,
    count: c_ulonglong,
    datatype: c_uint,
    peer: c_uint,
    comm: *const c_void,
    stream: *const c_void,
) -> c_int;

type NcclRecv = unsafe extern "C" fn(
    recvbuff: *mut c_void,
    count: c_ulonglong,
    datatype: c_uint,
    peer: c_uint,
    comm: *const c_void,
    stream: *const c_void,
) -> c_int;

// OnceCell for NCCL function pointers
static NCCL_ALL_REDUCE_FUNC: OnceCell<NcclAllReduce> = OnceCell::new();
static NCCL_BROADCAST_FUNC: OnceCell<NcclBroadcast> = OnceCell::new();
static NCCL_REDUCE_FUNC: OnceCell<NcclReduce> = OnceCell::new();
static NCCL_ALL_GATHER_FUNC: OnceCell<NcclAllGather> = OnceCell::new();
static NCCL_REDUCE_SCATTER_FUNC: OnceCell<NcclReduceScatter> = OnceCell::new();
static NCCL_ALL_TO_ALL_FUNC: OnceCell<NcclAlltoAll> = OnceCell::new();
static NCCL_GATHER_FUNC: OnceCell<NcclGather> = OnceCell::new();
static NCCL_SCATTER_FUNC: OnceCell<NcclScatter> = OnceCell::new();
static NCCL_SEND_FUNC: OnceCell<NcclSend> = OnceCell::new();
static NCCL_RECV_FUNC: OnceCell<NcclRecv> = OnceCell::new();

static NCCL_FUNCS_INIT_ONCE: Once = Once::new();

fn init_nccl_funcs() {
    NCCL_FUNCS_INIT_ONCE.call_once(|| {
        init();

        // Initialize each NCCL function separately to avoid type mismatch issues
        init_nccl_function("ncclAllReduce", &NCCL_ALL_REDUCE_FUNC);
        init_nccl_function("ncclBroadcast", &NCCL_BROADCAST_FUNC);
        init_nccl_function("ncclReduce", &NCCL_REDUCE_FUNC);
        init_nccl_function("ncclAllGather", &NCCL_ALL_GATHER_FUNC);
        init_nccl_function("ncclReduceScatter", &NCCL_REDUCE_SCATTER_FUNC);
        init_nccl_function("ncclAlltoAll", &NCCL_ALL_TO_ALL_FUNC);
        init_nccl_function("ncclGather", &NCCL_GATHER_FUNC);
        init_nccl_function("ncclScatter", &NCCL_SCATTER_FUNC);
        init_nccl_function("ncclSend", &NCCL_SEND_FUNC);
        init_nccl_function("ncclRecv", &NCCL_RECV_FUNC);
    });
}

fn init_nccl_function<T: Copy>(func_name: &str, func_cell: &OnceCell<T>) {
    debug!("Attempting to load NCCL function: {}", func_name);
    
    // 1. 尝试 RTLD_NEXT
    let c_func_name = CString::new(func_name).unwrap();
    let fn_ptr = unsafe { libc::dlsym(libc::RTLD_NEXT, c_func_name.as_ptr()) };
    
    if !fn_ptr.is_null() {
        debug!("Successfully loaded via RTLD_NEXT: {}", func_name);
        unsafe {
            let func: T = std::mem::transmute_copy(&fn_ptr);
            let _ = func_cell.set(func);
        }
        return;
    }
    
    // 2. 直接扫描 /proc/self/maps
    warn!("[HangDetect] RTLD_NEXT failed for {}, scanning /proc/self/maps for NCCL symbols", func_name);
    
    let maps = match File::open("/proc/self/maps") {
        Ok(f) => f,
        Err(e) => {
            error!("[HangDetect] Could not open /proc/self/maps while resolving {}: {}", func_name, e);
            panic!("Failed to open /proc/self/maps");
        }
    };

    let reader = BufReader::new(maps);
    for line in reader.lines().flatten() {
        // libnccl.so 或 libtorch_cuda.so
        if line.contains("libnccl.so") || line.contains("libtorch_cuda.so") {
            if let Some(path_start) = line.find('/') {
                let path = &line[path_start..].trim();

                debug!("[HangDetect] Inspecting library: {}", path);

                // 使用 RTLD_LAZY | RTLD_LOCAL 打开库
                let lib_res = unsafe { 
                    libloading::os::unix::Library::open(
                        Some(path),
                        libloading::os::unix::RTLD_LAZY | libloading::os::unix::RTLD_LOCAL
                    ) 
                };

                if let Ok(lib) = lib_res {
                    // Box::leak 不能释放
                    let lib_ref = Box::leak(Box::new(lib));

                    let sym_res = unsafe { lib_ref.get::<T>(c_func_name.as_bytes_with_nul()) };
                    
                    if let Ok(sym) = sym_res {
                        debug!("[HangDetect] Resolved NCCL symbol {} from {}", func_name, path);
                        let func: T = *sym; 
                        func_cell.set(func).unwrap_or_else(|_| {
                        });
                        return;
                    }
                }
            }
        }
    }

    // 3. 最终失败处理
    error!("CRITICAL: Failed to locate NCCL symbol '{}' in any loaded library", func_name);
    panic!("CRITICAL: Failed to locate symbol '{}' in any loaded library.", func_name);
}

#[derive(Debug)]
pub struct NCCLError {
    pub code: c_int,
}

impl std::fmt::Display for NCCLError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "NCCL error code: {}", self.code)
    }
}

// NCCL function wrappers
pub fn nccl_all_reduce(
    sendbuff: *const c_void,
    recvbuff: *mut c_void,
    count: c_ulonglong,
    datatype: c_uint,
    op: c_uint,
    comm: *const c_void,
    stream: *const c_void,
) -> Result<(), NCCLError> {
    init_nccl_funcs();
    let func = NCCL_ALL_REDUCE_FUNC.get().expect("NCCL_ALL_REDUCE_FUNC not initialized");
    let nccl_status = unsafe { func(sendbuff, recvbuff, count, datatype, op, comm, stream) };
    if nccl_status != 0 {
        Err(NCCLError { code: nccl_status })
    } else {
        Ok(())
    }
}

pub fn nccl_broadcast(
    sendbuff: *const c_void,
    recvbuff: *mut c_void,
    count: c_ulonglong,
    datatype: c_uint,
    root: c_uint,
    comm: *const c_void,
    stream: *const c_void,
) -> Result<(), NCCLError> {
    init_nccl_funcs();
    let func = NCCL_BROADCAST_FUNC.get().expect("NCCL_BROADCAST_FUNC not initialized");
    let nccl_status = unsafe { func(sendbuff, recvbuff, count, datatype, root, comm, stream) };
    if nccl_status != 0 {
        Err(NCCLError { code: nccl_status })
    } else {
        Ok(())
    }
}

// Similar wrappers for other NCCL functions...
pub fn nccl_reduce(
    sendbuff: *const c_void,
    recvbuff: *mut c_void,
    count: c_ulonglong,
    datatype: c_uint,
    op: c_uint,
    root: c_uint,
    comm: *const c_void,
    stream: *const c_void,
) -> Result<(), NCCLError> {
    init_nccl_funcs();
    let func = NCCL_REDUCE_FUNC.get().expect("NCCL_REDUCE_FUNC not initialized");
    let nccl_status = unsafe { func(sendbuff, recvbuff, count, datatype, op, root, comm, stream) };
    if nccl_status != 0 {
        Err(NCCLError { code: nccl_status })
    } else {
        Ok(())
    }
}

pub fn nccl_all_gather(
    sendbuff: *const c_void,
    recvbuff: *mut c_void,
    sendcount: c_ulonglong,
    datatype: c_uint,
    comm: *const c_void,
    stream: *const c_void,
) -> Result<(), NCCLError> {
    init_nccl_funcs();
    let func = NCCL_ALL_GATHER_FUNC.get().expect("NCCL_ALL_GATHER_FUNC not initialized");
    let nccl_status = unsafe { func(sendbuff, recvbuff, sendcount, datatype, comm, stream) };
    if nccl_status != 0 {
        Err(NCCLError { code: nccl_status })
    } else {
        Ok(())
    }
}

pub fn nccl_reduce_scatter(
    sendbuff: *const c_void,
    recvbuff: *mut c_void,
    recvcount: c_ulonglong,
    datatype: c_uint,
    op: c_uint,
    comm: *const c_void,
    stream: *const c_void,
) -> Result<(), NCCLError> {
    init_nccl_funcs();
    let func = NCCL_REDUCE_SCATTER_FUNC.get().expect("NCCL_REDUCE_SCATTER_FUNC not initialized");
    let nccl_status = unsafe { func(sendbuff, recvbuff, recvcount, datatype, op, comm, stream) };
    if nccl_status != 0 {
        Err(NCCLError { code: nccl_status })
    } else {
        Ok(())
    }
}

pub fn nccl_alltoall(
    sendbuff: *const c_void,
    recvbuff: *mut c_void,
    count: c_ulonglong,
    datatype: c_uint,
    comm: *const c_void,
    stream: *const c_void,
) -> Result<(), NCCLError> {
    init_nccl_funcs();
    let func = NCCL_ALL_TO_ALL_FUNC.get().expect("NCCL_ALL_TO_ALL_FUNC not initialized");
    let nccl_status = unsafe { func(sendbuff, recvbuff, count, datatype, comm, stream) };
    if nccl_status != 0 {
        Err(NCCLError { code: nccl_status })
    } else {
        Ok(())
    }
}

pub fn nccl_gather(
    sendbuff: *const c_void,
    recvbuff: *mut c_void,
    sendcount: c_ulonglong,
    datatype: c_uint,
    root: c_uint,
    comm: *const c_void,
    stream: *const c_void,
) -> Result<(), NCCLError> {
    init_nccl_funcs();
    let func = NCCL_GATHER_FUNC.get().expect("NCCL_GATHER_FUNC not initialized");
    let nccl_status = unsafe { func(sendbuff, recvbuff, sendcount, datatype, root, comm, stream) };
    if nccl_status != 0 {
        Err(NCCLError { code: nccl_status })
    } else {
        Ok(())
    }
}

pub fn nccl_scatter(
    sendbuff: *const c_void,
    recvbuff: *mut c_void,
    recvcount: c_ulonglong,
    datatype: c_uint,
    root: c_uint,
    comm: *const c_void,
    stream: *const c_void,
) -> Result<(), NCCLError> {
    init_nccl_funcs();
    let func = NCCL_SCATTER_FUNC.get().expect("NCCL_SCATTER_FUNC not initialized");
    let nccl_status = unsafe { func(sendbuff, recvbuff, recvcount, datatype, root, comm, stream) };
    if nccl_status != 0 {
        Err(NCCLError { code: nccl_status })
    } else {
        Ok(())
    }
}

pub fn nccl_send(
    sendbuff: *const c_void,
    count: c_ulonglong,
    datatype: c_uint,
    peer: c_uint,
    comm: *const c_void,
    stream: *const c_void,
) -> Result<(), NCCLError> {
    init_nccl_funcs();
    let func = NCCL_SEND_FUNC.get().expect("NCCL_SEND_FUNC not initialized");
    let nccl_status = unsafe { func(sendbuff, count, datatype, peer, comm, stream) };
    if nccl_status != 0 {
        Err(NCCLError { code: nccl_status })
    } else {
        Ok(())
    }
}

pub fn nccl_recv(
    recvbuff: *mut c_void,
    count: c_ulonglong,
    datatype: c_uint,
    peer: c_uint,
    comm: *const c_void,
    stream: *const c_void,
) -> Result<(), NCCLError> {
    init_nccl_funcs();
    let func = NCCL_RECV_FUNC.get().expect("NCCL_RECV_FUNC not initialized");
    let nccl_status = unsafe { func(recvbuff, count, datatype, peer, comm, stream) };
    if nccl_status != 0 {
        Err(NCCLError { code: nccl_status })
    } else {
        Ok(())
    }
}
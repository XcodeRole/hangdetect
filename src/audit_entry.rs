use libc::{
    c_char, 
    c_uint, 
    uintptr_t,
    Elf64_Sym, 
    Lmid_t,
    Elf64_Addr,
};
use std::ffi::CStr;
use std::sync::Once;

mod launch_wrappers;
mod cuda_funcs;
mod logger;
mod monitor;
use crate::logger::init_logger;

use crate::cuda_funcs::{RUNTIME_API, DRIVER_API, NCCL_API};

// const LAV_CURRENT: c_uint = 2;
const LA_FLG_BINDTO: c_uint = 0x00000001;
const LA_FLG_BINDFROM: c_uint = 0x00000002;
const LA_SYMB_NOPLTENTER: c_uint = 0x01;
const LA_SYMB_NOPLTEXIT: c_uint = 0x02;
// Dynamic Section
#[repr(C)]
pub struct Elf64_Dyn {
    pub d_tag: i64,  // Elf64_Sxword
    pub d_un: Elf64_Dyn_Union,
}

#[repr(C)]
pub union Elf64_Dyn_Union {
    pub d_val: u64,
    pub d_ptr: u64,
}

// Dynamic Tags
pub const DT_NULL:   i64 = 0;
pub const DT_HASH:   i64 = 4;
pub const DT_STRTAB: i64 = 5;
pub const DT_SYMTAB: i64 = 6;
pub const DT_SYMENT: i64 = 11;
#[repr(C)]
pub struct link_map {
    pub l_addr: Elf64_Addr,
    pub l_name: *mut c_char,
    pub l_ld: *mut Elf64_Dyn,
    pub l_next: *mut link_map,
    pub l_prev: *mut link_map,
}

static RUNTIME_INIT: Once = Once::new();
static DRIVER_INIT: Once = Once::new();
static NCCL_INIT: Once = Once::new();

fn init_runtime_from_map(def_map: *mut link_map) {
    RUNTIME_INIT.call_once(|| unsafe {
        let resolve = |sym: &str| -> Option<usize> {
            find_symbol_in_link_map(def_map, sym)
        };
        macro_rules! init_runtime_api {
            ($func_name:expr, $table_field:ident) => {
                if let Some(addr) = resolve($func_name) {
                    RUNTIME_API.$table_field = Some(std::mem::transmute(addr));
                } else {
                    panic!("[hangdetect][audit][rt-init] failed to resolve {}", $func_name);
                }
            };
        }
        init_runtime_api!("cudaFuncGetName", get_name);
        init_runtime_api!("cudaLaunchKernel", launch_kernel);
        init_runtime_api!("cudaLaunchKernelExC", launch_kernel_ex_c);
        init_runtime_api!("cudaStreamGetId", stream_get_id);
        init_runtime_api!("cudaEventCreateWithFlags", event_create_with_flags);
        init_runtime_api!("cudaEventDestroy", event_destroy);
        init_runtime_api!("cudaEventRecord", event_record);
        init_runtime_api!("cudaEventElapsedTime", event_elapsed_time);
        init_runtime_api!("cudaEventQuery", event_query);
    });
}

fn init_driver_from_map(def_map: *mut link_map) {
    DRIVER_INIT.call_once(|| unsafe {
        let resolve = |sym: &str| -> Option<usize> {
            find_symbol_in_link_map(def_map, sym)
        };
        macro_rules! init_driver_api {
            ($func_name:expr, $table_field:ident) => {
                if let Some(addr) = resolve($func_name) {
                    DRIVER_API.$table_field = Some(std::mem::transmute(addr));
                } else {
                    panic!("[hangdetect][audit][drv-init] failed to resolve {}", $func_name);
                }
            };
        }
        init_driver_api!("cuLaunchKernel", launch_kernel);
        init_driver_api!("cuLaunchKernelEx", launch_kernel_ex);
        init_driver_api!("cuFuncGetName", get_name);
    });
}

#[unsafe(no_mangle)]
pub extern "C" fn la_version(version: c_uint) -> c_uint {
    init_logger();
    version
}

#[unsafe(no_mangle)]
pub extern "C" fn la_objopen(
    map: *mut link_map,
    _lmid: Lmid_t,
    cookie: *mut uintptr_t,
) -> c_uint {
    unsafe {
        if !cookie.is_null() {
            *cookie = map as uintptr_t;
        }
    }
    LA_FLG_BINDFROM | LA_FLG_BINDTO
}

#[unsafe(no_mangle)]
pub extern "C" fn la_symbind64(
    sym: *mut Elf64_Sym,
    _ndx: c_uint,
    _refcook: *mut uintptr_t,
    defcook: *mut uintptr_t,
    flags: *mut c_uint,
    symname: *const c_char,
) -> uintptr_t {
    unsafe {
        if symname.is_null() {
            if !sym.is_null() {
                return (*sym).st_value as uintptr_t;
            } else {
                return 0;
            }
        }

        let c_str = CStr::from_ptr(symname);
        let name = c_str.to_str().unwrap_or("<non-utf8>");

        let is_target =
            name.contains("cuda") || name.starts_with("cu") || name.starts_with("nccl");

        if !is_target {
            if !sym.is_null() {
                return (*sym).st_value as uintptr_t;
            } else {
                return 0;
            }
        }

        let def_map = if !defcook.is_null() && *defcook != 0 {
            *defcook as *mut link_map
        } else {
            std::ptr::null_mut()
        };

        macro_rules! intercept_cuda_launch {
            ($func_name:ident) => {
                if name == stringify!($func_name) {
                    if !def_map.is_null() {
                        init_runtime_from_map(def_map);
                    }
                    *flags = LA_SYMB_NOPLTENTER | LA_SYMB_NOPLTEXIT;
                    return crate::launch_wrappers::$func_name as uintptr_t;
                }
            };
        }
        // 1. Intercept Runtime API
        intercept_cuda_launch!(cudaLaunchKernel);
        intercept_cuda_launch!(cudaLaunchKernelExC);
        // if name == "cudaLaunchKernel" {
        //     if !def_map.is_null() {
        //         init_runtime_from_map(def_map);
        //     }
        //     *flags = LA_SYMB_NOPLTENTER | LA_SYMB_NOPLTEXIT;
        //     return crate::launch_wrappers::cudaLaunchKernel as uintptr_t;
        // }
        
        // if name == "cudaLaunchKernelExC" {
        //     if !def_map.is_null() {
        //         init_runtime_from_map(def_map);
        //     }
        //     *flags = LA_SYMB_NOPLTENTER | LA_SYMB_NOPLTEXIT;
        //     return crate::launch_wrappers::cudaLaunchKernelExC as uintptr_t;
        // }

        macro_rules! intercept_cu_launch {
            ($func_name:ident) => {
                if name == stringify!($func_name) {
                    if !def_map.is_null() {
                        init_driver_from_map(def_map);
                    }
                    *flags = LA_SYMB_NOPLTENTER | LA_SYMB_NOPLTEXIT;
                    return crate::launch_wrappers::$func_name as uintptr_t;
                }
            };
        }
        // 2. Intercept Driver API
        intercept_cu_launch!(cuLaunchKernel);
        intercept_cu_launch!(cuLaunchKernelEx);
        // if name == "cuLaunchKernel" {
        //     if !def_map.is_null() {
        //         init_driver_from_map(def_map);
        //     }
        //     *flags = LA_SYMB_NOPLTENTER | LA_SYMB_NOPLTEXIT;
        //     return crate::launch_wrappers::cuLaunchKernel as uintptr_t;
        // }

        // if name == "cuLaunchKernelEx" {
        //     if !def_map.is_null() {
        //         init_driver_from_map(def_map);
        //     }
        //     *flags = LA_SYMB_NOPLTENTER | LA_SYMB_NOPLTEXIT;
        //     return crate::launch_wrappers::cuLaunchKernelEx as uintptr_t;
        // }

        // 3. Intercept NCCL API
        macro_rules! intercept_nccl {
            ($func_name:ident, $api_field:ident) => {
                if name == stringify!($func_name) {
                    if !sym.is_null() {
                        let addr = (*sym).st_value as usize;
                        NCCL_API.$api_field = Some(std::mem::transmute(addr));
                    }
                    *flags = LA_SYMB_NOPLTENTER | LA_SYMB_NOPLTEXIT;
                    return crate::launch_wrappers::$func_name as uintptr_t;
                }
            };
        }

        intercept_nccl!(ncclAllReduce, all_reduce);
        intercept_nccl!(ncclBroadcast, broadcast);
        intercept_nccl!(ncclBcast, bcast);
        intercept_nccl!(ncclReduce, reduce);
        intercept_nccl!(ncclAllGather, all_gather);
        intercept_nccl!(ncclReduceScatter, reduce_scatter);
        intercept_nccl!(ncclAlltoAll, all_to_all);
        intercept_nccl!(ncclGather, gather);
        intercept_nccl!(ncclScatter, scatter);
        intercept_nccl!(ncclSend, send);
        intercept_nccl!(ncclRecv, recv);

        // Default behavior: return original symbol value
        if !sym.is_null() {
            (*sym).st_value as uintptr_t
        } else {
            0
        }
    }
}

pub fn find_symbol_in_link_map(map: *mut link_map, target_name: &str) -> Option<usize> {
    unsafe {
        if map.is_null() || (*map).l_ld.is_null() {
            return None;
        }

        let l_addr = (*map).l_addr as usize;

        let so_name = if !(*map).l_name.is_null() {
            CStr::from_ptr((*map).l_name)
                .to_string_lossy()
                .into_owned()
        } else {
            "<null>".to_string()
        };

        let dyn_section = (*map).l_ld;

        let mut strtab_ptr: *const c_char = std::ptr::null();
        let mut symtab_ptr: *const Elf64_Sym = std::ptr::null();
        let mut hash_ptr: *const u32 = std::ptr::null(); // DT_HASH
        let mut syment_size: usize = std::mem::size_of::<Elf64_Sym>();

        // 1. 遍历 .dynamic，找 DT_STRTAB / DT_SYMTAB / DT_HASH / DT_SYMENT
        let mut curr_dyn = dyn_section;
        loop {
            let tag = (*curr_dyn).d_tag;
            if tag == DT_NULL {
                break;
            }

            match tag {
                DT_STRTAB => {
                    strtab_ptr = (*curr_dyn).d_un.d_ptr as *const c_char;
                }
                DT_SYMTAB => {
                    symtab_ptr = (*curr_dyn).d_un.d_ptr as *const Elf64_Sym;
                }
                DT_HASH => {
                    hash_ptr = (*curr_dyn).d_un.d_ptr as *const u32;
                }
                DT_SYMENT => {
                    syment_size = (*curr_dyn).d_un.d_val as usize;
                }
                _ => {}
            }

            curr_dyn = curr_dyn.add(1);
        }

        if strtab_ptr.is_null() || symtab_ptr.is_null() {
            eprintln!(
                "[hangdetect][audit][find_symbol] ERROR: missing DT_STRTAB or DT_SYMTAB in map='{}'",
                so_name
            );
            return None;
        }

        // 2. 通过 DT_HASH 获取符号数量
        let mut sym_count: usize = 0;

        if !hash_ptr.is_null() {
            // DT_HASH 布局：
            // [0] nbucket
            // [1] nchain
            // [2..2+nbucket)    bucket[]
            // [2+nbucket..]     chain[]
            let nbucket = *hash_ptr.add(0) as usize;
            let nchain = *hash_ptr.add(1) as usize;

            if nbucket == 0 || nchain == 0 {
                eprintln!(
                    "[hangdetect][audit][find_symbol] WARN: invalid DT_HASH in map='{}'",
                    so_name
                );
            } else {
                sym_count = nchain;
            }
        }

        if sym_count == 0 {
            sym_count = 8192;
        }

        // 3. 扫描符号表（0..sym_count），按名字匹配
        let target_bytes = target_name.as_bytes();

        for i in 0..sym_count {
            let sym = &*symtab_ptr.add(i);

            if sym.st_name == 0 {
                continue;
            }
            if syment_size != std::mem::size_of::<Elf64_Sym>() {
                eprintln!(
                    "[hangdetect][audit][find_symbol] ERROR: unexpected DT_SYMENT={} for map='{}'",
                    syment_size, so_name
                );
                return None;
            }

            let sym_name_ptr = strtab_ptr.add(sym.st_name as usize);

            let c_str = CStr::from_ptr(sym_name_ptr);
            if c_str.to_bytes() == target_bytes {
                let real_addr = l_addr.wrapping_add(sym.st_value as usize);
                return Some(real_addr);
            }
        }

        eprintln!(
            "[hangdetect][audit][find_symbol] ERROR: not found symbol '{}' in map='{}'",
            target_name, so_name
        );
        None
    }
}
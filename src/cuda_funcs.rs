use crate::init::init;
use libc::{c_int, c_uint, c_ulonglong, uintptr_t};
use std::ffi::c_void;
use std::ptr::null;
use std::sync::Once;

#[repr(C)]
pub struct Dim3 {
    pub x: u32,
    pub y: u32,
    pub z: u32,
}

type CudaFuncGetNameFunc = unsafe extern "C" fn(
    name: *mut *const std::ffi::c_char,
    func: *const c_void,
) -> std::ffi::c_int;

type CudaFuncLaunchKernel = unsafe extern "C" fn(
    func: *const c_void,
    grid_dim: Dim3,
    block_dim: Dim3,
    args: *const *const c_void,
    shared_mem: usize,
    stream: *mut c_void,
) -> std::ffi::c_int;

// /**
//  * CUDA extensible launch configuration
//  */
// typedef __device_builtin__ struct cudaLaunchConfig_st {
//     dim3 gridDim;               /**< Grid dimensions */
//     dim3 blockDim;              /**< Block dimensions */
//     size_t dynamicSmemBytes;    /**< Dynamic shared-memory size per thread block in bytes */
//     cudaStream_t stream;        /**< Stream identifier */
//     cudaLaunchAttribute *attrs; /**< List of attributes; nullable if ::cudaLaunchConfig_t::numAttrs == 0 */
//     unsigned int numAttrs;      /**< Number of attributes populated in ::cudaLaunchConfig_t::attrs */
// } cudaLaunchConfig_t;

#[repr(C)]
pub struct CudaLaunchConfig {
    grid_dim: Dim3,
    block_dim: Dim3,
    dynamic_smem_bytes: usize,
    pub stream: *mut c_void,
    attrs: *mut c_void,
    num_attrs: c_uint,
}

// cudaError_t cudaLaunchKernelExC ( const cudaLaunchConfig_t* config, const void* func, void** args )
type CudaFuncLaunchKernelExC = unsafe extern "C" fn(
    config: *const CudaLaunchConfig,
    func: *const c_void,
    args: *mut *const c_void,
) -> std::ffi::c_int;

// cudaError_t cudaStreamGetId ( cudaStream_t hStream, unsigned long long* streamId )
type CudaStreamGetId =
    unsafe extern "C" fn(stream: *const c_void, stream_id: *mut c_ulonglong) -> std::ffi::c_int;

// cudaError_t 	cudaEventCreateWithFlags ( cudaEvent_t* event, unsigned int  flags )
type CudaEventCreateWithFlags =
    unsafe extern "C" fn(event: *mut *const c_void, flags: c_uint) -> std::ffi::c_int;

// cudaError_t 	cudaEventDestroy ( cudaEvent_t event )
type CudaEventDestroy = unsafe extern "C" fn(event: *const c_void) -> std::ffi::c_int;

// cudaError_t 	cudaEventRecord ( cudaEvent_t event, cudaStream_t stream = 0 )
type CudaEventRecord =
    unsafe extern "C" fn(event: *const c_void, stream: *const c_void) -> std::ffi::c_int;

// cudaError_t cudaEventElapsedTime ( float* ms, cudaEvent_t start, cudaEvent_t end )
type CudaEventElapsedTime =
    unsafe extern "C" fn(ms: *mut f32, start: *const c_void, end: *const c_void) -> std::ffi::c_int;

// cudaError_t cudaEventQuery ( cudaEvent_t event )
type CudaEventQuery = unsafe extern "C" fn(event: *const c_void) -> std::ffi::c_int;

type CudaEventSynchronize = unsafe extern "C" fn(event: *const c_void) -> std::ffi::c_int;

// CUresult cuLaunchKernel ( CUfunction f, unsigned int  gridDimX, unsigned int  gridDimY,
//                           unsigned int  gridDimZ, unsigned int  blockDimX,
//                           unsigned int  blockDimY, unsigned int  blockDimZ,
//                           unsigned int  sharedMemBytes, CUstream hStream,
//                           void** kernelParams, void** extra )
type CuFuncLaunchKernel = unsafe extern "C" fn(
    func: *const c_void,
    grid_dim_x: c_uint,
    grid_dim_y: c_uint,
    grid_dim_z: c_uint,
    block_dim_x: c_uint,
    block_dim_y: c_uint,
    block_dim_z: c_uint,
    shared_mem: c_uint,
    stream: *const c_void,
    kernel_params: *mut *const c_void,
    extra: *mut *const c_void,
) -> std::ffi::c_int;

// /**
//  * CUDA extensible launch configuration
//  */
// typedef struct CUlaunchConfig_st {
//     unsigned int gridDimX;       /**< Width of grid in blocks */
//     unsigned int gridDimY;       /**< Height of grid in blocks */
//     unsigned int gridDimZ;       /**< Depth of grid in blocks */
//     unsigned int blockDimX;      /**< X dimension of each thread block */
//     unsigned int blockDimY;      /**< Y dimension of each thread block */
//     unsigned int blockDimZ;      /**< Z dimension of each thread block */
//     unsigned int sharedMemBytes; /**< Dynamic shared-memory size per thread block in bytes */
//     CUstream hStream;            /**< Stream identifier */
//     CUlaunchAttribute *attrs;    /**< List of attributes; nullable if ::CUlaunchConfig::numAttrs == 0 */
//     unsigned int numAttrs;       /**< Number of attributes populated in ::CUlaunchConfig::attrs */
// } CUlaunchConfig;

#[repr(C)]
pub struct CuLaunchConfig {
    grid_dim_x: c_uint,
    grid_dim_y: c_uint,
    grid_dim_z: c_uint,
    block_dim_x: c_uint,
    block_dim_y: c_uint,
    block_dim_z: c_uint,
    shared_mem_bytes: c_uint,
    pub stream: *const c_void,
    attrs: *mut c_void,
    num_attrs: c_uint,
}

// CUresult cuLaunchKernelEx ( const CUlaunchConfig* config, CUfunction f, void** kernelParams, void** extra)
type CuFuncLaunchKernelEx = unsafe extern "C" fn(
    config: *const CuLaunchConfig,
    func: *const c_void,
    args: *mut *const c_void,
) -> std::ffi::c_int;

// CUresult cuFuncGetName ( const char** name, CUfunction hfunc )
type CuFuncGetName = unsafe extern "C" fn(
    name: *mut *const std::ffi::c_char,
    func: *const c_void,
) -> std::ffi::c_int;

static CUDA_FUNCS_INIT_ONCE: Once = Once::new();
static mut CUDA_GET_NAME_FUNC: Option<CudaFuncGetNameFunc> = None;
static mut CUDA_LAUNCH_KERNEL_FUNC: Option<CudaFuncLaunchKernel> = None;

static mut CUDA_LAUNCH_KERNEL_EXC_FUNC: Option<CudaFuncLaunchKernelExC> = None;

static mut CUDA_STREAM_GET_ID_FUNC: Option<CudaStreamGetId> = None;

static mut CUDA_EVENT_CREATE_WITH_FLAGS_FUNC: Option<CudaEventCreateWithFlags> = None;

static mut CUDA_EVENT_DESTROY_FUNC: Option<CudaEventDestroy> = None;

static mut CUDA_EVENT_RECORD_FUNC: Option<CudaEventRecord> = None;
static mut CUDA_EVENT_ELAPSED_TIME_FUNC: Option<CudaEventElapsedTime> = None;

static mut CUDA_EVENT_QUERY_FUNC: Option<CudaEventQuery> = None;
static mut CUDA_EVENT_SYNCHRONIZE_FUNC: Option<CudaEventSynchronize> = None;

static CU_FUNCS_INIT_ONCE: Once = Once::new();
static mut CU_LAUNCH_KERNEL_FUNC: Option<CuFuncLaunchKernel> = None;

static mut CU_LAUNCH_KERNEL_EXC_FUNC: Option<CuFuncLaunchKernelEx> = None;

static mut CU_GET_NAME_FUNC: Option<CuFuncGetName> = None;

fn init_cuda_funcs() {
    CUDA_FUNCS_INIT_ONCE.call_once(|| unsafe {
        init();

        let sym = std::ffi::CString::new("cudaFuncGetName").unwrap();

        let fn_ptr = libc::dlsym(libc::RTLD_NEXT, sym.as_ptr());
        if fn_ptr.is_null() {
            panic!("failed to load cudaFuncGetName")
        }
        CUDA_GET_NAME_FUNC = Some(std::mem::transmute(fn_ptr));

        let sym = std::ffi::CString::new("cudaLaunchKernel").unwrap();
        let fn_ptr = libc::dlsym(libc::RTLD_NEXT, sym.as_ptr());
        if fn_ptr.is_null() {
            panic!("failed to load cudaLaunchKernel")
        }
        CUDA_LAUNCH_KERNEL_FUNC = Some(std::mem::transmute(fn_ptr));

        let sym = std::ffi::CString::new("cudaLaunchKernelExC").unwrap();
        let fn_ptr = libc::dlsym(libc::RTLD_NEXT, sym.as_ptr());
        if fn_ptr.is_null() {
            panic!("failed to load cudaLaunchKernelExC")
        }
        CUDA_LAUNCH_KERNEL_EXC_FUNC = Some(std::mem::transmute(fn_ptr));

        let sym = std::ffi::CString::new("cudaStreamGetId").unwrap();
        let fn_ptr = libc::dlsym(libc::RTLD_NEXT, sym.as_ptr());
        if fn_ptr.is_null() {
            panic!("failed to load cudaStreamGetId")
        }
        CUDA_STREAM_GET_ID_FUNC = Some(std::mem::transmute(fn_ptr));

        let sym = std::ffi::CString::new("cudaEventCreateWithFlags").unwrap();
        let fn_ptr = libc::dlsym(libc::RTLD_NEXT, sym.as_ptr());
        if fn_ptr.is_null() {
            panic!("failed to load cudaEventCreateWithFlags")
        }
        CUDA_EVENT_CREATE_WITH_FLAGS_FUNC = Some(std::mem::transmute(fn_ptr));

        let sym = std::ffi::CString::new("cudaEventDestroy").unwrap();
        let fn_ptr = libc::dlsym(libc::RTLD_NEXT, sym.as_ptr());
        if fn_ptr.is_null() {
            panic!("failed to load cudaEventDestroy")
        }
        CUDA_EVENT_DESTROY_FUNC = Some(std::mem::transmute(fn_ptr));

        let sym = std::ffi::CString::new("cudaEventRecord").unwrap();
        let fn_ptr = libc::dlsym(libc::RTLD_NEXT, sym.as_ptr());
        if fn_ptr.is_null() {
            panic!("failed to load cudaEventRecord")
        }
        CUDA_EVENT_RECORD_FUNC = Some(std::mem::transmute(fn_ptr));

        let sym = std::ffi::CString::new("cudaEventElapsedTime").unwrap();
        let fn_ptr = libc::dlsym(libc::RTLD_NEXT, sym.as_ptr());
        if fn_ptr.is_null() {
            panic!("failed to load cudaEventElapsedTime")
        }
        CUDA_EVENT_ELAPSED_TIME_FUNC = Some(std::mem::transmute(fn_ptr));

        let sym = std::ffi::CString::new("cudaEventQuery").unwrap();
        let fn_ptr = libc::dlsym(libc::RTLD_NEXT, sym.as_ptr());
        if fn_ptr.is_null() {
            panic!("failed to load cudaEventQuery")
        }
        CUDA_EVENT_QUERY_FUNC = Some(std::mem::transmute(fn_ptr));

        let sym = std::ffi::CString::new("cudaEventSynchronize").unwrap();
        let fn_ptr = libc::dlsym(libc::RTLD_NEXT, sym.as_ptr());
        if fn_ptr.is_null() {
            panic!("failed to load cudaEventSynchronize")
        }
        CUDA_EVENT_SYNCHRONIZE_FUNC = Some(std::mem::transmute(fn_ptr));
    })
}

fn init_cu_funcs() {
    CU_FUNCS_INIT_ONCE.call_once(|| unsafe {
        init();

        let sym = std::ffi::CString::new("cuLaunchKernel").unwrap();
        let fn_ptr = libc::dlsym(libc::RTLD_NEXT, sym.as_ptr());
        if fn_ptr.is_null() {
            panic!("failed to load cuLaunchKernel")
        }
        CU_LAUNCH_KERNEL_FUNC = Some(std::mem::transmute(fn_ptr));

        let sym = std::ffi::CString::new("cuLaunchKernelEx").unwrap();
        let fn_ptr = libc::dlsym(libc::RTLD_NEXT, sym.as_ptr());
        if fn_ptr.is_null() {
            panic!("failed to load cuLaunchKernelEx")
        }
        CU_LAUNCH_KERNEL_EXC_FUNC = Some(std::mem::transmute(fn_ptr));

        let sym = std::ffi::CString::new("cuFuncGetName").unwrap();
        let fn_ptr = libc::dlsym(libc::RTLD_NEXT, sym.as_ptr());
        if fn_ptr.is_null() {
            panic!("failed to load cuFuncGetName")
        }
        CU_GET_NAME_FUNC = Some(std::mem::transmute(fn_ptr));
    })
}

#[derive(Debug)]
pub struct CUDAError {
    pub code: c_int,
}
impl std::fmt::Display for CUDAError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "CUDA error code: {}", self.code)
    }
}

pub fn get_cuda_func_name(func: *const c_void) -> Result<String, CUDAError> {
    init_cuda_funcs();
    unsafe {
        let mut name_ptr: *const std::ffi::c_char = null();
        let cuda_status = CUDA_GET_NAME_FUNC.unwrap()(&mut name_ptr, func);
        if cuda_status != 0 {
            Err(CUDAError { code: cuda_status })
        } else {
            let cstr = std::ffi::CStr::from_ptr(name_ptr);
            Ok(cstr.to_str().unwrap().to_string())
        }
    }
}

pub fn launch_cuda_kernel(
    func: *const c_void,
    grid_dim: Dim3,
    block_dim: Dim3,
    args: *const *const c_void,
    shared_mem: usize,
    stream: *mut c_void,
) -> Result<(), CUDAError> {
    init_cuda_funcs();
    unsafe {
        let cuda_status =
            CUDA_LAUNCH_KERNEL_FUNC.unwrap()(func, grid_dim, block_dim, args, shared_mem, stream);
        if cuda_status != 0 {
            Err(CUDAError { code: cuda_status })
        } else {
            Ok(())
        }
    }
}

pub fn launch_cuda_kernel_ex_c(
    config: *const CudaLaunchConfig,
    func: *const c_void,
    args: *mut *const c_void,
) -> Result<(), CUDAError> {
    init_cuda_funcs();
    unsafe {
        let cuda_status = CUDA_LAUNCH_KERNEL_EXC_FUNC.unwrap()(config, func, args);
        if cuda_status != 0 {
            Err(CUDAError { code: cuda_status })
        } else {
            Ok(())
        }
    }
}

pub fn launch_cu_kernel(
    func: *const c_void,
    grid_dim_x: c_uint,
    grid_dim_y: c_uint,
    grid_dim_z: c_uint,
    block_dim_x: c_uint,
    block_dim_y: c_uint,
    block_dim_z: c_uint,
    shared_mem: c_uint,
    stream: *const c_void,
    kernel_params: *mut *const c_void,
    extra: *mut *const c_void,
) -> Result<(), CUDAError> {
    init_cu_funcs();
    unsafe {
        let cu_status = CU_LAUNCH_KERNEL_FUNC.unwrap()(
            func,
            grid_dim_x,
            grid_dim_y,
            grid_dim_z,
            block_dim_x,
            block_dim_y,
            block_dim_z,
            shared_mem,
            stream,
            kernel_params,
            extra,
        );
        if cu_status != 0 {
            Err(CUDAError { code: cu_status })
        } else {
            Ok(())
        }
    }
}

pub fn launch_cu_kernel_ex(
    config: *const CuLaunchConfig,
    func: *const c_void,
    args: *mut *const c_void,
) -> Result<(), CUDAError> {
    init_cu_funcs();
    unsafe {
        let cu_status = CU_LAUNCH_KERNEL_EXC_FUNC.unwrap()(config, func, args);
        if cu_status != 0 {
            Err(CUDAError { code: cu_status })
        } else {
            Ok(())
        }
    }
}

pub fn cu_func_get_name(func: *const c_void) -> Result<String, CUDAError> {
    init_cu_funcs();
    unsafe {
        let mut name_ptr: *const std::ffi::c_char = null();
        let cu_status = CU_GET_NAME_FUNC.unwrap()(&mut name_ptr, func);
        if cu_status != 0 {
            Err(CUDAError { code: cu_status })
        } else {
            let cstr = std::ffi::CStr::from_ptr(name_ptr);
            Ok(cstr.to_str().unwrap().to_string())
        }
    }
}

pub fn cuda_stream_get_id(stream: *const c_void) -> Result<u64, CUDAError> {
    init_cuda_funcs();
    unsafe {
        let mut stream_id: c_ulonglong = 0;
        let cuda_status = CUDA_STREAM_GET_ID_FUNC.unwrap()(stream, &mut stream_id);
        if cuda_status != 0 {
            Err(CUDAError { code: cuda_status })
        } else {
            Ok(stream_id as u64)
        }
    }
}

pub struct CUDAEvent {
    event: uintptr_t,
}

impl CUDAEvent {
    pub fn new() -> Result<CUDAEvent, CUDAError> {
        init_cuda_funcs();
        unsafe {
            let mut event: *const c_void = null();
            let cuda_status = CUDA_EVENT_CREATE_WITH_FLAGS_FUNC.unwrap()(&mut event, 0);
            if cuda_status != 0 {
                Err(CUDAError { code: cuda_status })
            } else {
                Ok(CUDAEvent {
                    event: event as uintptr_t,
                })
            }
        }
    }

    pub fn record(&self, stream: *const c_void) -> Result<(), CUDAError> {
        init_cuda_funcs();
        unsafe {
            let cuda_status = CUDA_EVENT_RECORD_FUNC.unwrap()(self.event as *const c_void, stream);
            if cuda_status != 0 {
                Err(CUDAError { code: cuda_status })
            } else {
                Ok(())
            }
        }
    }

    pub fn since(&self, begin: &CUDAEvent) -> Result<f32, CUDAError> {
        init_cuda_funcs();
        unsafe {
            let mut ms: f32 = 0.0;
            let cuda_status = CUDA_EVENT_ELAPSED_TIME_FUNC.unwrap()(
                &mut ms,
                begin.event as *const c_void,
                self.event as *const c_void,
            );
            if cuda_status != 0 {
                Err(CUDAError { code: cuda_status })
            } else {
                Ok(ms)
            }
        }
    }

    pub fn query(&self) -> Result<bool, CUDAError> {
        init_cuda_funcs();
        unsafe {
            let cuda_status = CUDA_EVENT_QUERY_FUNC.unwrap()(self.event as *const c_void);
            if cuda_status == 0 {
                Ok(true)
            } else if cuda_status == 600 {
                // cudaErrorNotReady
                Ok(false)
            } else {
                Err(CUDAError { code: cuda_status })
            }
        }
    }

    pub fn synchronize(&self) -> Result<(), CUDAError> {
        init_cuda_funcs();
        unsafe {
            let cuda_status = CUDA_EVENT_SYNCHRONIZE_FUNC.unwrap()(self.event as *const c_void);
            if cuda_status != 0 {
                Err(CUDAError { code: cuda_status })
            } else {
                Ok(())
            }
        }
    }
}

impl Drop for CUDAEvent {
    fn drop(&mut self) {
        init_cuda_funcs();
        unsafe {
            let cuda_status = CUDA_EVENT_DESTROY_FUNC.unwrap()(self.event as *const c_void);
            if cuda_status != 0 {
                eprintln!("failed to destroy CUDA event: {}", cuda_status);
            }
        }
    }
}

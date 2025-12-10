use libc::{c_int, c_uint, c_ulonglong, uintptr_t};
use std::ffi::c_void;
use std::ptr::null;

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
    kernel_params: *mut *const c_void,
    extra: *mut *const c_void,
) -> std::ffi::c_int;

// CUresult cuFuncGetName ( const char** name, CUfunction hfunc )
type CuFuncGetName = unsafe extern "C" fn(
    name: *mut *const std::ffi::c_char,
    func: *const c_void,
) -> std::ffi::c_int;

pub struct RuntimeApiTable {
    pub get_name: Option<CudaFuncGetNameFunc>,
    pub launch_kernel: Option<CudaFuncLaunchKernel>,
    pub launch_kernel_ex_c: Option<CudaFuncLaunchKernelExC>,
    pub stream_get_id: Option<CudaStreamGetId>,
    pub event_create_with_flags: Option<CudaEventCreateWithFlags>,
    pub event_destroy: Option<CudaEventDestroy>,
    pub event_record: Option<CudaEventRecord>,
    pub event_elapsed_time: Option<CudaEventElapsedTime>,
    pub event_query: Option<CudaEventQuery>,
    pub event_synchronize: Option<CudaEventSynchronize>,
}

pub struct DriverApiTable {
    pub launch_kernel: Option<CuFuncLaunchKernel>,
    pub launch_kernel_ex: Option<CuFuncLaunchKernelEx>,
    pub get_name: Option<CuFuncGetName>,
}

pub static mut RUNTIME_API: RuntimeApiTable = RuntimeApiTable {
    get_name: None,
    launch_kernel: None,
    launch_kernel_ex_c: None,
    stream_get_id: None,
    event_create_with_flags: None,
    event_destroy: None,
    event_record: None,
    event_elapsed_time: None,
    event_query: None,
    event_synchronize: None,
};

pub static mut DRIVER_API: DriverApiTable = DriverApiTable {
    launch_kernel: None,
    launch_kernel_ex: None,
    get_name: None,
};

// NCCL Types
pub type ncclComm_t = *mut c_void;
pub type ncclDataType_t = c_int;
pub type ncclRedOp_t = c_int;
pub type ncclResult_t = c_int;

// Function pointer types for NCCL
pub type NcclFuncAllReduce = unsafe extern "C" fn(
    sendbuff: *const c_void,
    recvbuff: *mut c_void,
    count: usize,
    datatype: ncclDataType_t,
    op: ncclRedOp_t,
    comm: ncclComm_t,
    stream: *mut c_void,
) -> ncclResult_t;

pub type NcclFuncBroadcast = unsafe extern "C" fn(
    sendbuff: *const c_void,
    recvbuff: *mut c_void,
    count: usize,
    datatype: ncclDataType_t,
    root: c_int,
    comm: ncclComm_t,
    stream: *mut c_void,
) -> ncclResult_t;

pub type NcclFuncBcast = unsafe extern "C" fn(
    buff: *mut c_void,
    count: usize,
    datatype: ncclDataType_t,
    root: c_int,
    comm: ncclComm_t,
    stream: *mut c_void,
) -> ncclResult_t;

pub type NcclFuncReduce = unsafe extern "C" fn(
    sendbuff: *const c_void,
    recvbuff: *mut c_void,
    count: usize,
    datatype: ncclDataType_t,
    op: ncclRedOp_t,
    root: c_int,
    comm: ncclComm_t,
    stream: *mut c_void,
) -> ncclResult_t;

pub type NcclFuncAllGather = unsafe extern "C" fn(
    sendbuff: *const c_void,
    recvbuff: *mut c_void,
    sendcount: usize,
    datatype: ncclDataType_t,
    comm: ncclComm_t,
    stream: *mut c_void,
) -> ncclResult_t;

pub type NcclFuncReduceScatter = unsafe extern "C" fn(
    sendbuff: *const c_void,
    recvbuff: *mut c_void,
    recvcount: usize,
    datatype: ncclDataType_t,
    op: ncclRedOp_t,
    comm: ncclComm_t,
    stream: *mut c_void,
) -> ncclResult_t;

pub type NcclFuncAllToAll = unsafe extern "C" fn(
    sendbuff: *const c_void,
    recvbuff: *mut c_void,
    count: usize,
    datatype: ncclDataType_t,
    comm: ncclComm_t,
    stream: *mut c_void,
) -> ncclResult_t;

pub type NcclFuncGather = unsafe extern "C" fn(
    sendbuff: *const c_void,
    recvbuff: *mut c_void,
    count: usize,
    datatype: ncclDataType_t,
    root: c_int,
    comm: ncclComm_t,
    stream: *mut c_void,
) -> ncclResult_t;

pub type NcclFuncScatter = unsafe extern "C" fn(
    sendbuff: *const c_void,
    recvbuff: *mut c_void,
    count: usize,
    datatype: ncclDataType_t,
    root: c_int,
    comm: ncclComm_t,
    stream: *mut c_void,
) -> ncclResult_t;

pub type NcclFuncSend = unsafe extern "C" fn(
    sendbuff: *const c_void,
    count: usize,
    datatype: ncclDataType_t,
    peer: c_int,
    comm: ncclComm_t,
    stream: *mut c_void,
) -> ncclResult_t;

pub type NcclFuncRecv = unsafe extern "C" fn(
    recvbuff: *mut c_void,
    count: usize,
    datatype: ncclDataType_t,
    peer: c_int,
    comm: ncclComm_t,
    stream: *mut c_void,
) -> ncclResult_t;

pub struct NcclApiTable {
    pub all_reduce: Option<NcclFuncAllReduce>,
    pub broadcast: Option<NcclFuncBroadcast>,
    pub bcast: Option<NcclFuncBcast>,
    pub reduce: Option<NcclFuncReduce>,
    pub all_gather: Option<NcclFuncAllGather>,
    pub reduce_scatter: Option<NcclFuncReduceScatter>,
    pub all_to_all: Option<NcclFuncAllToAll>,
    pub gather: Option<NcclFuncGather>,
    pub scatter: Option<NcclFuncScatter>,
    pub send: Option<NcclFuncSend>,
    pub recv: Option<NcclFuncRecv>,
}

pub static mut NCCL_API: NcclApiTable = NcclApiTable {
    all_reduce: None,
    broadcast: None,
    bcast: None,
    reduce: None,
    all_gather: None,
    reduce_scatter: None,
    all_to_all: None,
    gather: None,
    scatter: None,
    send: None,
    recv: None,
};

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
    unsafe {
        let func_ptr = match RUNTIME_API.get_name {
            Some(f) => f,
            None => {
                return Err(CUDAError { code: -1 });
            }
        };
        let mut name_ptr: *const std::ffi::c_char = null();
        let cuda_status = func_ptr(&mut name_ptr, func);
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
    unsafe {
        let func_ptr = match RUNTIME_API.launch_kernel {
            Some(f) => f,
            None => {
                return Err(CUDAError { code: -1 });
            }
        };
        let cuda_status = func_ptr(func, grid_dim, block_dim, args, shared_mem, stream);
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
    unsafe {
        let func_ptr = match RUNTIME_API.launch_kernel_ex_c {
            Some(f) => f,
            None => {
                return Err(CUDAError { code: -1 });
            }
        };
        let cuda_status = func_ptr(config, func, args);
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
    unsafe {
        let func_ptr = match DRIVER_API.launch_kernel {
            Some(f) => f,
            None => {
                return Err(CUDAError { code: -1 });
            }
        };
        let cu_status = func_ptr(
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
    kernel_params: *mut *const c_void,
    extra: *mut *const c_void,
) -> Result<(), CUDAError> {
    unsafe {
        let func_ptr = match DRIVER_API.launch_kernel_ex {
            Some(f) => f,
            None => {
                return Err(CUDAError { code: -1 });
            }
        };
        let cu_status = func_ptr(config, func, kernel_params, extra);
        if cu_status != 0 {
            Err(CUDAError { code: cu_status })
        } else {
            Ok(())
        }
    }
}

pub fn cu_func_get_name(func: *const c_void) -> Result<String, CUDAError> {
    unsafe {
        let func_ptr = match DRIVER_API.get_name {
            Some(f) => f,
            None => {
                return Err(CUDAError { code: -1 });
            }
        };
        let mut name_ptr: *const std::ffi::c_char = null();
        let cu_status = func_ptr(&mut name_ptr, func);
        if cu_status != 0 {
            log::warn!(
                "[hangdetect][cuda_funcs] cuFuncGetName failed for func={:p} with code={}",
                func,
                cu_status
            );
            Ok(format!("unknown_cu_func(0x{:p})", func))
        } else {
            let cstr = std::ffi::CStr::from_ptr(name_ptr);
            Ok(cstr.to_str().unwrap().to_string())
        }
    }
}

pub fn cuda_stream_get_id(stream: *const c_void) -> Result<u64, CUDAError> {
    unsafe {
        let func_ptr = match RUNTIME_API.stream_get_id {
            Some(f) => f,
            None => {
                return Err(CUDAError { code: -1 });
            }
        };
        let mut stream_id: c_ulonglong = 0;
        let cuda_status = func_ptr(stream, &mut stream_id);
        if cuda_status != 0 {
            Err(CUDAError { code: cuda_status })
        } else {
            Ok(stream_id as u64)
        }
    }
}

// NCCL Wrappers
pub fn nccl_all_reduce(
    sendbuff: *const c_void,
    recvbuff: *mut c_void,
    count: usize,
    datatype: ncclDataType_t,
    op: ncclRedOp_t,
    comm: ncclComm_t,
    stream: *mut c_void,
) -> ncclResult_t {
    unsafe {
        match NCCL_API.all_reduce {
            Some(f) => f(sendbuff, recvbuff, count, datatype, op, comm, stream),
            None => {
                log::error!("[hangdetect][nccl] ncclAllReduce not resolved");
                -1
            }
        }
    }
}

pub fn nccl_broadcast(
    sendbuff: *const c_void,
    recvbuff: *mut c_void,
    count: usize,
    datatype: ncclDataType_t,
    root: c_int,
    comm: ncclComm_t,
    stream: *mut c_void,
) -> ncclResult_t {
    unsafe {
        match NCCL_API.broadcast {
            Some(f) => f(sendbuff, recvbuff, count, datatype, root, comm, stream),
            None => {
                log::error!("[hangdetect][nccl] ncclBroadcast not resolved");
                -1
            }
        }
    }
}

pub fn nccl_bcast(
    buff: *mut c_void,
    count: usize,
    datatype: ncclDataType_t,
    root: c_int,
    comm: ncclComm_t,
    stream: *mut c_void,
) -> ncclResult_t {
    unsafe {
        match NCCL_API.bcast {
            Some(f) => f(buff, count, datatype, root, comm, stream),
            None => {
                log::error!("[hangdetect][nccl] ncclBcast not resolved");
                -1
            }
        }
    }
}

pub fn nccl_reduce(
    sendbuff: *const c_void,
    recvbuff: *mut c_void,
    count: usize,
    datatype: ncclDataType_t,
    op: ncclRedOp_t,
    root: c_int,
    comm: ncclComm_t,
    stream: *mut c_void,
) -> ncclResult_t {
    unsafe {
        match NCCL_API.reduce {
            Some(f) => f(sendbuff, recvbuff, count, datatype, op, root, comm, stream),
            None => {
                log::error!("[hangdetect][nccl] ncclReduce not resolved");
                -1
            }
        }
    }
}

pub fn nccl_all_gather(
    sendbuff: *const c_void,
    recvbuff: *mut c_void,
    sendcount: usize,
    datatype: ncclDataType_t,
    comm: ncclComm_t,
    stream: *mut c_void,
) -> ncclResult_t {
    unsafe {
        match NCCL_API.all_gather {
            Some(f) => f(sendbuff, recvbuff, sendcount, datatype, comm, stream),
            None => {
                log::error!("[hangdetect][nccl] ncclAllGather not resolved");
                -1
            }
        }
    }
}

pub fn nccl_reduce_scatter(
    sendbuff: *const c_void,
    recvbuff: *mut c_void,
    recvcount: usize,
    datatype: ncclDataType_t,
    op: ncclRedOp_t,
    comm: ncclComm_t,
    stream: *mut c_void,
) -> ncclResult_t {
    unsafe {
        match NCCL_API.reduce_scatter {
            Some(f) => f(sendbuff, recvbuff, recvcount, datatype, op, comm, stream),
            None => {
                log::error!("[hangdetect][nccl] ncclReduceScatter not resolved");
                -1
            }
        }
    }
}

pub fn nccl_all_to_all(
    sendbuff: *const c_void,
    recvbuff: *mut c_void,
    count: usize,
    datatype: ncclDataType_t,
    comm: ncclComm_t,
    stream: *mut c_void,
) -> ncclResult_t {
    unsafe {
        match NCCL_API.all_to_all {
            Some(f) => f(sendbuff, recvbuff, count, datatype, comm, stream),
            None => {
                log::error!("[hangdetect][nccl] ncclAlltoAll not resolved");
                -1
            }
        }
    }
}

pub fn nccl_gather(
    sendbuff: *const c_void,
    recvbuff: *mut c_void,
    count: usize,
    datatype: ncclDataType_t,
    root: c_int,
    comm: ncclComm_t,
    stream: *mut c_void,
) -> ncclResult_t {
    unsafe {
        match NCCL_API.gather {
            Some(f) => f(sendbuff, recvbuff, count, datatype, root, comm, stream),
            None => {
                log::error!("[hangdetect][nccl] ncclGather not resolved");
                -1
            }
        }
    }
}

pub fn nccl_scatter(
    sendbuff: *const c_void,
    recvbuff: *mut c_void,
    count: usize,
    datatype: ncclDataType_t,
    root: c_int,
    comm: ncclComm_t,
    stream: *mut c_void,
) -> ncclResult_t {
    unsafe {
        match NCCL_API.scatter {
            Some(f) => f(sendbuff, recvbuff, count, datatype, root, comm, stream),
            None => {
                log::error!("[hangdetect][nccl] ncclScatter not resolved");
                -1
            }
        }
    }
}

pub fn nccl_send(
    sendbuff: *const c_void,
    count: usize,
    datatype: ncclDataType_t,
    peer: c_int,
    comm: ncclComm_t,
    stream: *mut c_void,
) -> ncclResult_t {
    unsafe {
        match NCCL_API.send {
            Some(f) => f(sendbuff, count, datatype, peer, comm, stream),
            None => {
                log::error!("[hangdetect][nccl] ncclSend not resolved");
                -1
            }
        }
    }
}

pub fn nccl_recv(
    recvbuff: *mut c_void,
    count: usize,
    datatype: ncclDataType_t,
    peer: c_int,
    comm: ncclComm_t,
    stream: *mut c_void,
) -> ncclResult_t {
    unsafe {
        match NCCL_API.recv {
            Some(f) => f(recvbuff, count, datatype, peer, comm, stream),
            None => {
                log::error!("[hangdetect][nccl] ncclRecv not resolved");
                -1
            }
        }
    }
}

pub struct CUDAEvent {
    event: uintptr_t,
}

impl CUDAEvent {
    pub fn new() -> Result<CUDAEvent, CUDAError> {
        unsafe {
            let func_ptr = match RUNTIME_API.event_create_with_flags {
                Some(f) => f,
                None => {
                    return Err(CUDAError { code: -1 });
                }
            };
            let mut event: *const c_void = null();
            let cuda_status = func_ptr(&mut event, 0);
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
        unsafe {
            let func_ptr = match RUNTIME_API.event_record {
                Some(f) => f,
                None => {
                    return Err(CUDAError { code: -1 });
                }
            };
            let cuda_status = func_ptr(self.event as *const c_void, stream);
            if cuda_status != 0 {
                Err(CUDAError { code: cuda_status })
            } else {
                Ok(())
            }
        }
    }

    pub fn since(&self, begin: &CUDAEvent) -> Result<f32, CUDAError> {
        unsafe {
            let func_ptr = match RUNTIME_API.event_elapsed_time {
                Some(f) => f,
                None => {
                    return Err(CUDAError { code: -1 });
                }
            };
            let mut ms: f32 = 0.0;
            let cuda_status = func_ptr(
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
        unsafe {
            let func_ptr = match RUNTIME_API.event_query {
                Some(f) => f,
                None => {
                    return Err(CUDAError { code: -1 });
                }
            };
            let cuda_status = func_ptr(self.event as *const c_void);
            if cuda_status == 0 {
                Ok(true)
            } else if cuda_status == 600 {
                Ok(false)
            } else {
                Err(CUDAError { code: cuda_status })
            }
        }
    }
    pub fn synchronize(&self) -> Result<(), CUDAError> {
        unsafe {
            let func_ptr = match RUNTIME_API.event_synchronize {
                Some(f) => f,
                None => {
                    return Err(CUDAError { code: -1 });
                }
            };
            loop {
                let cuda_status = func_ptr(self.event as *const c_void);
                if cuda_status == 0 {
                    return Ok(());
                } else if cuda_status == 600 {
                    continue;
                } else {
                    return Err(CUDAError { code: cuda_status });
                }
            }
        }
    }
}

impl Drop for CUDAEvent {
    fn drop(&mut self) {
        unsafe {
            if let Some(func_ptr) = RUNTIME_API.event_destroy {
                let cuda_status = func_ptr(self.event as *const c_void);
                if cuda_status != 0 {
                    eprintln!("failed to destroy CUDA event: {}", cuda_status);
                }
            } else {
                eprintln!(
                    "[hangdetect][cuda_funcs] CUDA_EVENT_DESTROY_FUNC is not initialized; skip destroying CUDA event",
                );
            }
        }
    }
}

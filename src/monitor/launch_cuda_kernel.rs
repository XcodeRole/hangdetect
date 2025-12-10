use crate::cuda_funcs::cuda_stream_get_id;
use crate::monitor::error::MonitorError;
use anyhow::Context;
use cpp_demangle::Symbol;
use libc::uintptr_t;
use once_cell::sync::Lazy;
use std::collections::HashMap;
use std::ffi::c_void;
use std::fmt::{Display, Formatter};
use std::sync::{Arc, RwLock};

pub enum LaunchCUDAKernel {
    Runtime {
        func: *const c_void,
        stream: *const c_void,
    },

    Driver {
        func: *const c_void,
        stream: *const c_void,
    },

    Nccl {
        name: &'static str,
        stream: *const c_void,
    },
}

pub struct FuncName {
    symbol: String,
    demangled: Option<String>,
}

impl FuncName {
    pub fn display_name(&self) -> &str {
        if let Some(demangled) = &self.demangled {
            demangled
        } else {
            &self.symbol
        }
    }
}

trait GetKernelName {
    fn get_name(&self, func: *const c_void) -> Result<Arc<FuncName>, MonitorError>;
}

struct KernelNameCache<F>
where
    F: Fn(*const c_void) -> Result<String, crate::cuda_funcs::CUDAError>,
{
    cache: Arc<RwLock<HashMap<uintptr_t, Arc<FuncName>>>>,
    get_name_func: F,
}

fn new_kernel_name_cache<F>(f: F) -> KernelNameCache<F>
where
    F: Fn(*const c_void) -> Result<String, crate::cuda_funcs::CUDAError>,
{
    KernelNameCache {
        cache: Arc::new(RwLock::new(HashMap::new())),
        get_name_func: f,
    }
}

impl<F> GetKernelName for KernelNameCache<F>
where
    F: Fn(*const c_void) -> Result<String, crate::cuda_funcs::CUDAError>,
{
    fn get_name(&self, func: *const c_void) -> Result<Arc<FuncName>, MonitorError> {
        let func_ptr = func as uintptr_t;
        {
            let cache_read = self.cache.read().unwrap();
            if let Some(name) = cache_read.get(&func_ptr) {
                return Ok(name.clone());
            }
        }

        let mut cache_write = self.cache.write().unwrap();
        if let Some(name) = cache_write.get(&func_ptr) {
            return Ok(name.clone());
        }

        let name = (self.get_name_func)(func).map_err(MonitorError::CUDAError)?;
        let symbol = match Symbol::new(&name)
            .with_context(|| format!("symbol new error {}", name))
            .and_then(|s| {
                s.demangle()
                    .with_context(|| format!("demangle error {}", name))
            }) {
            Ok(demangled) => Some(demangled),
            Err(err) => {
                log::warn!("failed to demangle symbol {}: {}", name, err);
                None
            }
        };

        let f_name = Arc::new(FuncName {
            symbol: name,
            demangled: symbol,
        });

        cache_write.insert(func_ptr, f_name.clone());
        Ok(f_name)
    }
}

static RUNTIME_KERNEL_NAME_LOOKUP_FN: Lazy<
    Box<dyn Fn(*const c_void) -> Result<Arc<FuncName>, MonitorError> + Sync + Send>,
> = Lazy::new(|| {
    let cache = new_kernel_name_cache(crate::cuda_funcs::get_cuda_func_name);
    Box::new(move |func: *const c_void| cache.get_name(func))
});

static DRIVER_KERNEL_NAME_LOOKUP_FN: Lazy<
    Box<dyn Fn(*const c_void) -> Result<Arc<FuncName>, MonitorError> + Sync + Send>,
> = Lazy::new(|| {
    let cache = new_kernel_name_cache(crate::cuda_funcs::cu_func_get_name);
    Box::new(move |func: *const c_void| cache.get_name(func))
});

impl LaunchCUDAKernel {
    pub fn func_name(&self) -> Result<Arc<FuncName>, MonitorError> {
        match self {
            LaunchCUDAKernel::Runtime { func, .. } => (RUNTIME_KERNEL_NAME_LOOKUP_FN)(*func),
            LaunchCUDAKernel::Driver { func, .. } => (DRIVER_KERNEL_NAME_LOOKUP_FN)(*func),
            LaunchCUDAKernel::Nccl { name, .. } => Ok(Arc::new(FuncName {
                symbol: name.to_string(),
                demangled: None,
            })),
        }
    }

    pub fn stream(&self) -> *const c_void {
        match self {
            LaunchCUDAKernel::Runtime { stream, .. } => *stream,
            LaunchCUDAKernel::Driver { stream, .. } => *stream,
            LaunchCUDAKernel::Nccl { stream, .. } => *stream,
        }
    }
    pub fn stream_id(&self) -> Result<u64, MonitorError> {
        cuda_stream_get_id(self.stream()).map_err(MonitorError::CUDAError)
    }
}

impl Display for LaunchCUDAKernel {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "<{} Kernel: {} on stream {}>",
            match self {
                LaunchCUDAKernel::Runtime { .. } => "Runtime",
                LaunchCUDAKernel::Driver { .. } => "Driver",
                LaunchCUDAKernel::Nccl { .. } => "NCCL",
            },
            self.func_name()
                .map_err(|_| std::fmt::Error)?
                .display_name(),
            self.stream_id().map_err(|_| std::fmt::Error)?,
        )
    }
}

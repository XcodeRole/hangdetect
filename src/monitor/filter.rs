use super::monitor_aspect::{MonitorAspect, Operation};
use regex::Regex;
use std::env;
use std::sync::OnceLock;

pub trait Filter: Send + Sync {
    fn filter(&self, op: &Operation<'_>) -> bool;
}

pub fn merge_filter<F, A>(f: F, other: A) -> AspectWithBlock<A, F>
where
    A: MonitorAspect,
    F: Filter,
{
    AspectWithBlock {
        aspect: other,
        filter: f,
    }
}

pub struct AspectWithBlock<A, F>
where
    A: MonitorAspect,
    F: Filter,
{
    aspect: A,
    filter: F,
}

impl<A, B> MonitorAspect for AspectWithBlock<A, B>
where
    A: MonitorAspect,
    B: Filter,
{
    fn before_call(&self, op: &Operation<'_>) -> Result<(), crate::monitor::error::MonitorError> {
        if self.filter.filter(op) {
            self.aspect.before_call(op)
        } else {
            Ok(())
        }
    }

    fn after_call(&self, op: &Operation<'_>) -> Result<(), crate::monitor::error::MonitorError> {
        if self.filter.filter(op) {
            self.aspect.after_call(op)
        } else {
            Ok(())
        }
    }
}

pub struct KernelNameFilter {
    regex: Option<Regex>,
}

impl KernelNameFilter {
    pub fn new() -> Self {
        static REGEX: OnceLock<Option<Regex>> = OnceLock::new();
        
        let regex = REGEX.get_or_init(|| {
            match env::var("HANGDETECT_KERNEL_FILTER") {
                Ok(pattern) => {
                    match Regex::new(&pattern) {
                        Ok(regex) => {
                            log::info!("Kernel filter regex initialized with pattern: {}", pattern);
                            Some(regex)
                        }
                        Err(e) => {
                            log::warn!("Invalid kernel filter regex pattern '{}': {}", pattern, e);
                            None
                        }
                    }
                }
                Err(_) => {
                    log::info!("No kernel filter regex specified, all kernels will be enabled");
                    None
                }
            }
        }).clone();
        
        Self { regex }
    }
}

impl Filter for KernelNameFilter {
    fn filter(&self, op: &Operation<'_>) -> bool {
        match &self.regex {
            Some(regex) => match op {
                Operation::LaunchCUDAKernel(launch) => {
                    match launch.func_name() {
                        Ok(func_name) => {
                            let name = func_name.display_name();
                            let matches = regex.is_match(name);
                            if !matches {
                                log::debug!("Kernel '{}' filtered out by regex", name);
                            }
                            matches
                        }
                        Err(_) => {
                            // If we can't get the kernel name, disallow it by default
                            false
                        }
                    }
                }
                Operation::NCCLCommunication(comm) => {
                    let name = comm.api_name();
                    let matches = regex.is_match(name);
                    if !matches {
                        log::debug!("NCCL '{}' filtered out by regex", name);
                    }
                    matches
                }
            },
            None => {
                // No regex specified, allow all operations
                true
            }
        }
    }
}

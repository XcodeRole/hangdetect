use super::filter::merge_filter;
use super::filter::KernelNameFilter;
use super::logging_aspect::LoggingAspect;
use super::monitor_aspect::{MonitorAspect, Operation};
use crate::monitor::kernel_exec_time_aspect::KernelExecTimeAspect;
use crate::monitor::thread_local_enabler::ThreadLocalEnabler;
use once_cell::sync::Lazy;

struct MergeAspects<A, B>
where
    A: MonitorAspect,
    B: MonitorAspect,
{
    aspect_a: A,
    aspect_b: B,
}

impl<A, B> MonitorAspect for MergeAspects<A, B>
where
    A: MonitorAspect,
    B: MonitorAspect,
{
    fn before_call(&self, op: &Operation<'_>) -> Result<(), crate::monitor::error::MonitorError> {
        self.aspect_a.before_call(op)?;
        self.aspect_b.before_call(op)
    }

    fn after_call(&self, op: &Operation<'_>) -> Result<(), crate::monitor::error::MonitorError> {
        self.aspect_a.after_call(op)?;
        self.aspect_b.after_call(op)
    }
}

fn merge_aspect<A, B>(a: A, b: B) -> MergeAspects<A, B>
where
    A: MonitorAspect,
    B: MonitorAspect,
{
    MergeAspects {
        aspect_a: a,
        aspect_b: b,
    }
}

pub static ASPECTS: Lazy<Box<dyn MonitorAspect + Send + Sync>> = Lazy::new(|| {
    // Add more aspects here
    let aspect = LoggingAspect {};
    let aspect = merge_aspect(aspect, KernelExecTimeAspect);
    
    // Add kernel name filter
    let kernel_filter = KernelNameFilter::new();
    let merged_aspect = merge_filter(kernel_filter, aspect);
    
    // Add thread local enabler
    let merged_aspect = merge_filter(ThreadLocalEnabler {}, merged_aspect);
    
    Box::new(merged_aspect) as Box<dyn MonitorAspect + Send + Sync>
});

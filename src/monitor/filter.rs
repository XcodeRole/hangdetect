use super::monitor_aspect::MonitorAspect;
use crate::monitor::LaunchCUDAKernel;

pub trait Filter: Send + Sync {
    fn filter(&self, launch: &LaunchCUDAKernel) -> bool;
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
    fn before_call(
        &self,
        launch: &LaunchCUDAKernel,
    ) -> Result<(), crate::monitor::error::MonitorError> {
        if self.filter.filter(launch) {
            self.aspect.before_call(launch)
        } else {
            Ok(())
        }
    }

    fn after_call(
        &self,
        launch: &LaunchCUDAKernel,
    ) -> Result<(), crate::monitor::error::MonitorError> {
        if self.filter.filter(launch) {
            self.aspect.after_call(launch)
        } else {
            Ok(())
        }
    }
}

use ash::vk;

/// Cherry-picked from wgpu-hal crate

#[derive(Clone)]
pub struct RelaySemaphores {
    pub wait: Option<vk::Semaphore>,
    pub signal: vk::Semaphore,
}

impl RelaySemaphores {
    pub fn new(device: &ash::Device) -> Self {
        Self {
            wait: None,
            signal: unsafe { device.create_semaphore(&vk::SemaphoreCreateInfo::default(), None).unwrap() },
        }
    }
    
    pub fn advance(&mut self, device: &ash::Device) -> Self {
        let old = self.clone();
        match self.wait {
            None => {
                self.wait = Some(old.signal);
                self.signal = unsafe { device.create_semaphore(&vk::SemaphoreCreateInfo::default(), None).unwrap() }
            }
            Some(ref mut wait) => {
                std::mem::swap(wait, &mut self.signal);
            }
        }
        old
    }

    pub fn destroy(&self, device: &ash::Device) {
        if let Some(wait) = self.wait {
            unsafe { device.destroy_semaphore(wait, None) };
        }
        unsafe { device.destroy_semaphore(self.signal, None) };
    }
}

#[derive(Default)]
pub struct TimelineFence {
    pub last_completed: u64,
    active: Vec<(u64, vk::Fence)>,
    free: Vec<vk::Fence>,
}

impl TimelineFence {
    pub fn check_latest(&self, device: &ash::Device) -> u64 {
        let mut latest = self.last_completed;
        for &(value, fence) in self.active.iter() {
            if value <= latest {
                continue;
            }
            if unsafe { device.get_fence_status(fence).unwrap() } {
                latest = value;
            }
        }
        latest
    }

    pub fn wait_for(&self, device: &ash::Device, wait_value: u64) {
        if wait_value <= self.last_completed {
            return;
        }
        if let Some(&(_, fence)) = self.active.iter().find(|x| x.0 >= wait_value) {
            unsafe { device.wait_for_fences(&[fence], true, u64::MAX).unwrap() };
        }
    }

    pub fn maintain(&mut self, device: &ash::Device) {
        let latest = self.check_latest(device);
        let base_free = self.free.len();
        for &(value, fence) in self.active.iter() {
            if value <= latest {
                self.free.push(fence);
            }
        }
        if base_free != self.free.len() {
            self.active.retain(|&(value, _)| value > latest);
            unsafe { device.reset_fences(&self.free[base_free..]).unwrap() };
        }
        self.last_completed = latest;
    }

    pub fn add(&mut self, device: &ash::Device, value: u64) -> vk::Fence {
        let fence = self
            .free
            .pop()
            .unwrap_or_else(|| unsafe { device.create_fence(&vk::FenceCreateInfo::default(), None).unwrap() });
        self.active.push((value, fence));
        fence
    }

    pub fn destroy(&mut self, device: &ash::Device) {
        for &(_, fence) in self.active.iter() {
            unsafe { device.destroy_fence(fence, None) };
        }
        for &fence in self.free.iter() {
            unsafe { device.destroy_fence(fence, None) };
        }
    }
}

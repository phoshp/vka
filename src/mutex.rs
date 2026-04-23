use std::cell::UnsafeCell;
use std::ops::Deref;
use std::ops::DerefMut;

use parking_lot::lock_api::RawMutex;

/// A simple posix-like mutex that uses parking_lot's RawMutex under the hood.
pub struct Rutex<T> {
    data: UnsafeCell<T>,
    raw: parking_lot::RawMutex,
}

impl <T> Rutex<T> {
    pub fn new(data: T) -> Self {
        Self {
            data: UnsafeCell::new(data),
            raw: parking_lot::RawMutex::INIT,
        }
    }

    pub fn lock(&self) {
        self.raw.lock();
    }

    pub fn unlock(&self) {
        unsafe {
            self.raw.unlock();
        }
    }

    pub fn is_locked(&self) -> bool {
        self.raw.is_locked()
    }
}

impl<T> Deref for Rutex<T> {
    type Target = T;
    fn deref(&self) -> &Self::Target {
        unsafe { &*self.data.get() }
    }
}

impl<T> DerefMut for Rutex<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { &mut *self.data.get() }
    }
}

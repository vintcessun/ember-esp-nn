#![no_std]

mod ops;
mod quant;

pub use ops::EspBackend;

#[no_mangle]
pub extern "C" fn puts(_s: *const core::ffi::c_char) -> core::ffi::c_int {
    0
}

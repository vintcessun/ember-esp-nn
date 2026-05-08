#![no_std]

#[allow(non_camel_case_types)]
#[allow(non_snake_case)]
#[allow(non_upper_case_globals)]
#[allow(improper_ctypes)]
#[allow(clippy::missing_safety_doc)]
#[allow(clippy::too_many_arguments)]
pub mod bindings {
    include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
}

pub use bindings::*;

mod add;
mod conv2d;
mod depthwise;
mod fully_connected;
mod mul;
mod pool;
mod softmax;

pub struct EspBackend;

use ember_infer_core::{
    Conv2dParams, DepthwiseConv2dParams, ElementwiseAddParams, FullyConnectedParams, KernelBackend,
    MulParams, PoolParams, SoftmaxParams, Status,
};

/// Required base alignment (in bytes) for scratch buffers handed to ESP-NN.
///
/// The ESP32-S3 ESP-NN kernels cast the scratch buffer to `int16_t*` and issue
/// 128-bit PIE loads/stores (`ee.vld.128` / `ee.vst.128`) against it, and their
/// internal sub-buffer offset math (e.g.
/// `input_data16 = scratch + filter_size + align_len`) only aligns the
/// *offset*, assuming the base is already 16-byte aligned. On ESP32-S3 an
/// aligned SIMD load simply *truncates* the low address bits, so a misaligned
/// scratch base is read shifted — silently producing wrong results instead of
/// faulting.
///
/// A `Vec<u8>` / `&mut [u8]` scratch buffer is only 1-byte aligned, so whether
/// its base happens to land on a 16-byte boundary depends on the heap layout,
/// which differs between debug and release builds. That is exactly why the same
/// model inferred correctly in `release` but produced garbage in `debug`.
///
/// Each scratch-using op reserves `SCRATCH_ALIGN` extra bytes in its
/// `*_scratch_size` and bumps the base up to a 16-byte boundary via
/// [`aligned_scratch_ptr`] before calling the ESP-NN `set_*_scratch_buf`.
pub(crate) const SCRATCH_ALIGN: usize = 16;

/// Bump the base pointer of `scratch` up to a [`SCRATCH_ALIGN`]-byte boundary.
///
/// Callers must have sized the buffer with `+ SCRATCH_ALIGN` slack (see each
/// op's `scratch_size`) so the returned window still covers the size ESP-NN
/// asked for.
#[inline]
pub(crate) fn aligned_scratch_ptr(scratch: &mut [u8]) -> *mut core::ffi::c_void {
    let base = scratch.as_mut_ptr();
    let offset = base.align_offset(SCRATCH_ALIGN);
    // Guard against a degenerate/too-small slice: never step past the end.
    if offset == usize::MAX || offset >= scratch.len() {
        return base.cast::<core::ffi::c_void>();
    }
    // SAFETY: `offset < scratch.len()`, so the resulting pointer is in bounds.
    unsafe { base.add(offset).cast::<core::ffi::c_void>() }
}

#[cfg(all(feature = "trace-ops", debug_assertions))]
macro_rules! trace_op {
    ($($arg:tt)*) => {
        defmt::info!($($arg)*)
    };
}

#[cfg(not(all(feature = "trace-ops", debug_assertions)))]
macro_rules! trace_op {
    ($($arg:tt)*) => {};
}

impl KernelBackend for EspBackend {
    fn conv2d(&mut self, params: Conv2dParams<'_>) -> Status {
        trace_op!("[ember_esp_nn] enter conv2d");
        let status = conv2d::run(params);
        trace_op!("[ember_esp_nn] exit conv2d ok={}", status.is_ok());
        status
    }

    fn depthwise_conv2d(&mut self, params: DepthwiseConv2dParams<'_>) -> Status {
        trace_op!("[ember_esp_nn] enter depthwise");
        let status = depthwise::run(params);
        trace_op!("[ember_esp_nn] exit depthwise ok={}", status.is_ok());
        status
    }

    fn fully_connected(&mut self, params: FullyConnectedParams<'_>) -> Status {
        trace_op!("[ember_esp_nn] enter fully_connected");
        let status = fully_connected::run(params);
        trace_op!("[ember_esp_nn] exit fully_connected ok={}", status.is_ok());
        status
    }

    fn avg_pool(&mut self, params: PoolParams<'_>) -> Status {
        trace_op!("[ember_esp_nn] enter avg_pool");
        let status = pool::run_avg(params);
        trace_op!("[ember_esp_nn] exit avg_pool ok={}", status.is_ok());
        status
    }

    fn max_pool(&mut self, params: PoolParams<'_>) -> Status {
        trace_op!("[ember_esp_nn] enter max_pool");
        let status = pool::run_max(params);
        trace_op!("[ember_esp_nn] exit max_pool ok={}", status.is_ok());
        status
    }

    fn softmax(&mut self, params: SoftmaxParams<'_>) -> Status {
        trace_op!("[ember_esp_nn] enter softmax");
        let status = softmax::run(params);
        trace_op!("[ember_esp_nn] exit softmax ok={}", status.is_ok());
        status
    }

    fn add(&mut self, params: ElementwiseAddParams<'_>) -> Status {
        trace_op!("[ember_esp_nn] enter add");
        let status = add::run(params);
        trace_op!("[ember_esp_nn] exit add ok={}", status.is_ok());
        status
    }

    fn mul(&mut self, params: MulParams<'_>) -> Status {
        trace_op!("[ember_esp_nn] enter mul");
        let status = mul::run(params);
        trace_op!("[ember_esp_nn] exit mul ok={}", status.is_ok());
        status
    }

    fn conv2d_scratch_size(
        input_shape: [usize; 4],
        weights_shape: [usize; 4],
        output_shape: [usize; 4],
    ) -> usize {
        conv2d::scratch_size(input_shape, weights_shape, output_shape)
    }

    fn depthwise_conv2d_scratch_size(
        input_shape: [usize; 4],
        weights_shape: [usize; 4],
        output_shape: [usize; 4],
    ) -> usize {
        depthwise::scratch_size(input_shape, weights_shape, output_shape)
    }

    fn softmax_scratch_size(num_classes: usize) -> usize {
        softmax::scratch_size(num_classes)
    }
}

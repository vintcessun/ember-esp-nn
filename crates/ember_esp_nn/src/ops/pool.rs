#[cfg(all(feature = "esp32s3", not(feature = "force-ansi-pool")))]
use esp_nn_sys::bindings::{
    esp_nn_avg_pool_s8_esp32s3 as esp_nn_avg_pool_s8,
    esp_nn_max_pool_s8_esp32s3 as esp_nn_max_pool_s8,
};
#[cfg(all(feature = "esp32p4", not(feature = "force-ansi-pool")))]
use esp_nn_sys::bindings::{
    esp_nn_avg_pool_s8_esp32p4 as esp_nn_avg_pool_s8,
    esp_nn_max_pool_s8_esp32p4 as esp_nn_max_pool_s8,
};
#[cfg(any(
    feature = "force-ansi-pool",
    not(any(feature = "esp32s3", feature = "esp32p4"))
))]
use esp_nn_sys::bindings::{
    esp_nn_avg_pool_s8_ansi as esp_nn_avg_pool_s8,
    esp_nn_max_pool_s8_ansi as esp_nn_max_pool_s8,
};

use ember_infer_core::{FusedActivation, KernelError, Padding, PoolParams, Status};

/// The ESP32-S3 pool kernels read the input with aligned SIMD loads, so a
/// non-16-byte-aligned input tensor is read shifted (silently wrong). The
/// `#[model]` macro aligns intermediate activations, but a model that feeds an
/// unaligned buffer (e.g. the model input) straight into a pool would still hit
/// this — route those to the alignment-agnostic ANSI kernel. On other targets
/// the primary kernel is alignment-agnostic, so this is always `true`.
#[cfg(all(feature = "esp32s3", not(feature = "force-ansi-pool")))]
#[inline]
fn pool_can_use_simd(input: *const i8) -> bool {
    (input as usize).is_multiple_of(16)
}
#[cfg(not(all(feature = "esp32s3", not(feature = "force-ansi-pool"))))]
#[inline]
fn pool_can_use_simd(_input: *const i8) -> bool {
    true
}

pub fn run_avg(params: PoolParams<'_>) -> Status {
    #[cfg(all(feature = "esp32s3", not(feature = "force-ansi-pool")))]
    if !pool_can_use_simd(params.input.as_ptr()) {
        return run(params, esp_nn_sys::bindings::esp_nn_avg_pool_s8_ansi);
    }
    run(params, esp_nn_avg_pool_s8)
}

pub fn run_max(params: PoolParams<'_>) -> Status {
    #[cfg(all(feature = "esp32s3", not(feature = "force-ansi-pool")))]
    {
        // esp32s3 assembly has a padding bug — fall back to ansi when pad > 0,
        // and it also requires a 16-byte aligned input.
        let (pad_w, pad_h) = match params.padding {
            Padding::Valid => (0i32, 0i32),
            Padding::Same => (
                compute_padding(
                    params.input_shape[2] as i32,
                    params.output_shape[2] as i32,
                    params.stride_w,
                    params.filter_w,
                ),
                compute_padding(
                    params.input_shape[1] as i32,
                    params.output_shape[1] as i32,
                    params.stride_h,
                    params.filter_h,
                ),
            ),
        };
        if pad_w > 0 || pad_h > 0 || !pool_can_use_simd(params.input.as_ptr()) {
            return run(params, esp_nn_sys::bindings::esp_nn_max_pool_s8_ansi);
        }
    }
    run(params, esp_nn_max_pool_s8)
}

type PoolKernel = unsafe fn(
    *const i8,
    u16,
    u16,
    *mut i8,
    u16,
    u16,
    u16,
    u16,
    u16,
    u16,
    u16,
    u16,
    i32,
    i32,
    u16,
);

fn run(params: PoolParams<'_>, kernel: PoolKernel) -> Status {
    if params.input_shape[0] != 1
        || params.output_shape[0] != 1
        || params.input_shape[3] != params.output_shape[3]
        || !pool_dims_fit_u16(&params)
        || tensor_len(params.input_shape).is_none_or(|len| params.input.len() < len)
        || tensor_len(params.output_shape).is_none_or(|len| params.output.len() < len)
    {
        return Err(KernelError::InvalidShape);
    }

    let (pad_w, pad_h) = match params.padding {
        Padding::Valid => (0, 0),
        Padding::Same => (
            compute_padding(
                params.input_shape[2] as i32,
                params.output_shape[2] as i32,
                params.stride_w,
                params.filter_w,
            ),
            compute_padding(
                params.input_shape[1] as i32,
                params.output_shape[1] as i32,
                params.stride_h,
                params.filter_h,
            ),
        ),
    };
    let (activation_min, activation_max) = activation_range(params.activation, params.output_quant);

    unsafe {
        kernel(
            params.input.as_ptr(),
            params.input_shape[2] as u16,
            params.input_shape[1] as u16,
            params.output.as_mut_ptr(),
            params.output_shape[2] as u16,
            params.output_shape[1] as u16,
            params.stride_w as u16,
            params.stride_h as u16,
            params.filter_w as u16,
            params.filter_h as u16,
            pad_w as u16,
            pad_h as u16,
            activation_min,
            activation_max,
            params.input_shape[3] as u16,
        );
    }

    Ok(())
}

#[inline]
fn pool_dims_fit_u16(params: &PoolParams<'_>) -> bool {
    params.input_shape[1] <= u16::MAX as usize
        && params.input_shape[2] <= u16::MAX as usize
        && params.input_shape[3] <= u16::MAX as usize
        && params.output_shape[1] <= u16::MAX as usize
        && params.output_shape[2] <= u16::MAX as usize
        && params.stride_w > 0
        && params.stride_h > 0
        && params.filter_w > 0
        && params.filter_h > 0
        && params.stride_w <= u16::MAX as i32
        && params.stride_h <= u16::MAX as i32
        && params.filter_w <= u16::MAX as i32
        && params.filter_h <= u16::MAX as i32
}

#[inline]
fn tensor_len(shape: [usize; 4]) -> Option<usize> {
    shape
        .into_iter()
        .try_fold(1usize, |acc, dim| acc.checked_mul(dim))
}

#[inline]
fn compute_padding(input_dim: i32, output_dim: i32, stride: i32, filter: i32) -> i32 {
    ((output_dim - 1) * stride + filter - input_dim).max(0) / 2
}

#[inline]
fn activation_range(act: FusedActivation, output_quant: ember_infer_core::QuantParam) -> (i32, i32) {
    match act {
        FusedActivation::None => (-128, 127),
        FusedActivation::Relu => (output_quant.zero_point.max(-128), 127),
        FusedActivation::Relu6 => (
            output_quant.zero_point.max(-128),
            (output_quant.zero_point + round_f32_to_i32(6.0 / output_quant.scale)).min(127),
        ),
        FusedActivation::ReluN1To1 | FusedActivation::Tanh => (
            (output_quant.zero_point + round_f32_to_i32(-1.0 / output_quant.scale)).max(-128),
            (output_quant.zero_point + round_f32_to_i32(1.0 / output_quant.scale)).min(127),
        ),
        FusedActivation::Sigmoid => (
            output_quant.zero_point.max(-128),
            (output_quant.zero_point + round_f32_to_i32(1.0 / output_quant.scale)).min(127),
        ),
        FusedActivation::SignBit => (-128, 127),
    }
}

#[inline]
fn round_f32_to_i32(value: f32) -> i32 {
    libm::roundf(value) as i32
}

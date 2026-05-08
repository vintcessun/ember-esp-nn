#[cfg(not(any(feature = "esp32s3", feature = "esp32p4")))]
use esp_nn_sys::bindings::esp_nn_avg_pool_s8_ansi as esp_nn_avg_pool_s8;
#[cfg(any(feature = "esp32s3", feature = "esp32p4"))]
use esp_nn_sys::bindings::esp_nn_avg_pool_s8_esp32s3 as esp_nn_avg_pool_s8;

#[cfg(not(any(feature = "esp32s3", feature = "esp32p4")))]
use esp_nn_sys::bindings::esp_nn_max_pool_s8_ansi as esp_nn_max_pool_s8;
#[cfg(any(feature = "esp32s3", feature = "esp32p4"))]
use esp_nn_sys::bindings::esp_nn_max_pool_s8_esp32s3 as esp_nn_max_pool_s8;

use ember_infer_core::{FusedActivation, KernelError, Padding, PoolParams, Status};

pub fn run_avg(params: PoolParams<'_>) -> Status {
    run(params, esp_nn_avg_pool_s8)
}

pub fn run_max(params: PoolParams<'_>) -> Status {
    #[cfg(any(feature = "esp32s3", feature = "esp32p4"))]
    {
        // esp32s3 assembly has a padding bug — fall back to ansi when pad > 0
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
        if pad_w > 0 || pad_h > 0 {
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
    let (activation_min, activation_max) =
        activation_range(params.activation, params.output_quant.zero_point);

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
fn activation_range(act: FusedActivation, out_zero_point: i32) -> (i32, i32) {
    match act {
        FusedActivation::None => (-128, 127),
        FusedActivation::Relu | FusedActivation::Relu6 => (out_zero_point.max(-128), 127),
        _ => (-128, 127),
    }
}

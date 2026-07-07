#[cfg(all(feature = "esp32s3", not(feature = "force-ansi-depthwise")))]
use esp_nn_sys::bindings::{
    esp_nn_depthwise_conv_s8_esp32s3 as esp_nn_depthwise_conv_s8,
    esp_nn_get_depthwise_conv_scratch_size_esp32s3 as esp_nn_get_depthwise_conv_scratch_size,
    esp_nn_set_depthwise_conv_scratch_buf_esp32s3 as esp_nn_set_depthwise_conv_scratch_buf,
};
#[cfg(all(feature = "esp32p4", not(feature = "force-ansi-depthwise")))]
use esp_nn_sys::bindings::{
    esp_nn_depthwise_conv_s8_esp32p4 as esp_nn_depthwise_conv_s8,
    esp_nn_get_depthwise_conv_scratch_size_esp32p4 as esp_nn_get_depthwise_conv_scratch_size,
    esp_nn_set_depthwise_conv_scratch_buf_esp32p4 as esp_nn_set_depthwise_conv_scratch_buf,
};
#[cfg(any(
    feature = "force-ansi-depthwise",
    not(any(feature = "esp32s3", feature = "esp32p4"))
))]
use esp_nn_sys::bindings::{
    esp_nn_depthwise_conv_s8_ansi as esp_nn_depthwise_conv_s8,
    esp_nn_get_depthwise_conv_scratch_size_ansi as esp_nn_get_depthwise_conv_scratch_size,
    esp_nn_set_depthwise_conv_scratch_buf_ansi as esp_nn_set_depthwise_conv_scratch_buf,
};
// ESP32-S3 only: the optimized depthwise dispatcher feeds `input_data` to
// `esp_nn_aligned_s8_to_s16_with_offset` (an aligned `ee.vld.128`), so it
// silently produces garbage when the input tensor is not 16-byte aligned. The
// `#[model]` macro's activation tensors are only 1-byte aligned and the caller's
// scratch is linker-capped (no room to bounce a copy), so we route unaligned
// inputs to the alignment-agnostic ANSI kernel instead.
#[cfg(all(feature = "esp32s3", not(feature = "force-ansi-depthwise")))]
use esp_nn_sys::bindings::{
    esp_nn_depthwise_conv_s8_ansi, esp_nn_set_depthwise_conv_scratch_buf_ansi,
};

use crate::quant::quantize_multiplier;
use ember_infer_core::{DepthwiseConv2dParams, FusedActivation, KernelError, Padding, Status};
use esp_nn_sys::bindings::{act_params_t, data_2d_t, data_dims_t, dw_conv_params_t, quant_data_t};

// Compile-time stack limit for per-channel quantization. Raise this if a model
// needs more output channels and the target stack budget allows it.
const MAX_CHANNELS: usize = 512;
static mut DEPTHWISE_MULT: [i32; MAX_CHANNELS] = [0; MAX_CHANNELS];
static mut DEPTHWISE_SHIFT: [i32; MAX_CHANNELS] = [0; MAX_CHANNELS];

#[inline(never)]
fn invalid_shape<T>(message: &str) -> Result<T, KernelError> {
    #[cfg(debug_assertions)]
    panic!("{message}");
    #[cfg(not(debug_assertions))]
    {
        let _ = message;
        Err(KernelError::InvalidShape)
    }
}

pub fn run(params: DepthwiseConv2dParams<'_>) -> Status {
    if params.depth_multiplier <= 0 {
        return invalid_shape("depthwise invalid depth_multiplier");
    }

    let Some(c_out) = params.input_shape[3].checked_mul(params.depth_multiplier as usize) else {
        return invalid_shape("depthwise c_out overflow");
    };

    if c_out == 0
        || c_out > MAX_CHANNELS
        || c_out != params.output_shape[3]
        || params.weights_shape[3] != c_out
    {
        return invalid_shape("depthwise invalid channels");
    }

    if !shape4_fits_i32(params.input_shape)
        || !shape4_fits_i32(params.weights_shape)
        || !shape4_fits_i32(params.output_shape)
        || tensor_len(params.input_shape).is_none_or(|len| params.input.len() < len)
        || tensor_len(params.weights_shape).is_none_or(|len| params.weights.len() < len)
        || tensor_len(params.output_shape).is_none_or(|len| params.output.len() < len)
        || params.bias.is_some_and(|bias| bias.len() < c_out)
    {
        return invalid_shape("depthwise invalid tensor shape");
    }

    let (mult_arr, shift_arr) = depthwise_quant_buffers();
    for i in 0..c_out {
        let weight_scale = params
            .weights_per_channel_quant
            .and_then(|per_channel| per_channel.scales.get(i).copied())
            .unwrap_or(params.weights_quant.scale);
        let effective_scale =
            (params.input_quant.scale * weight_scale) / params.output_quant.scale;
        let (multiplier, shift) = quantize_multiplier(effective_scale as f64);
        mult_arr[i] = multiplier;
        shift_arr[i] = shift;
    }

    let input_dims = io_dims_from_nhwc(params.input_shape);
    let filter_dims = depthwise_filter_dims(params.weights_shape);
    let output_dims = io_dims_from_nhwc(params.output_shape);
    let (pad_w, pad_h) = match params.padding {
        Padding::Valid => (0, 0),
        Padding::Same => (
            compute_padding(
                params.input_shape[2] as i32,
                params.output_shape[2] as i32,
                params.stride_w,
                params.weights_shape[2] as i32,
            ),
            compute_padding(
                params.input_shape[1] as i32,
                params.output_shape[1] as i32,
                params.stride_h,
                params.weights_shape[1] as i32,
            ),
        ),
    };
    let (activation_min, activation_max) = activation_range(params.activation, params.output_quant);
    let dilation_w = normalize_esp_nn_dilation(params.dilation_w_factor);
    let dilation_h = normalize_esp_nn_dilation(params.dilation_h_factor);

    let dw_params = dw_conv_params_t {
        in_offset: -params.input_quant.zero_point,
        out_offset: params.output_quant.zero_point,
        ch_mult: params.depth_multiplier,
        stride: data_2d_t {
            width: params.stride_w,
            height: params.stride_h,
        },
        padding: data_2d_t {
            width: pad_w,
            height: pad_h,
        },
        dilation: data_2d_t {
            width: dilation_w,
            height: dilation_h,
        },
        activation: act_params_t {
            min: activation_min,
            max: activation_max,
        },
    };
    let q_data = quant_data_t {
        shift: shift_arr.as_mut_ptr(),
        mult: mult_arr.as_mut_ptr(),
    };

    #[cfg(all(feature = "trace-ops", debug_assertions))]
    defmt::info!(
        "[ember_esp_nn] depthwise params in={}x{}x{} filter={}x{} weights_c={} out={}x{}x{} ch_mult={} stride={}x{} pad={}x{} dilation={}x{} scratch={}",
        input_dims.width,
        input_dims.height,
        input_dims.channels,
        filter_dims.width,
        filter_dims.height,
        params.weights_shape[3],
        output_dims.width,
        output_dims.height,
        output_dims.channels,
        dw_params.ch_mult,
        dw_params.stride.width,
        dw_params.stride.height,
        dw_params.padding.width,
        dw_params.padding.height,
        dw_params.dilation.width,
        dw_params.dilation.height,
        params.scratch.len()
    );

    // The ESP32-S3 depthwise dispatcher feeds `input_data` to
    // `esp_nn_aligned_s8_to_s16_with_offset` (an aligned `ee.vld.128`), which
    // reads garbage unless the input is 16-byte aligned. The `#[model]` macro's
    // activation tensors are only 1-byte aligned and the caller's scratch is
    // linker-capped (no room to bounce an aligned copy for large models), so
    // when the input isn't aligned we run the correct, alignment-agnostic ANSI
    // kernel instead of the SIMD one. The scratch base is still 16-byte aligned
    // for the SIMD path's internal sub-buffers.
    let scratch_ptr = super::aligned_scratch_ptr(params.scratch);
    let input = params.input.as_ptr();
    let bias_ptr = params.bias.map_or(core::ptr::null(), |bias| bias.as_ptr());
    let output = params.output.as_mut_ptr();

    unsafe {
        if depthwise_can_use_simd(input) {
            esp_nn_set_depthwise_conv_scratch_buf(scratch_ptr);
            esp_nn_depthwise_conv_s8(
                &input_dims, input, &filter_dims, params.weights.as_ptr(), bias_ptr, &output_dims,
                output, &dw_params, &q_data,
            );
        } else {
            // Only reachable on ESP32-S3 (elsewhere `depthwise_can_use_simd` is
            // always true and the primary kernel is alignment-agnostic).
            #[cfg(all(feature = "esp32s3", not(feature = "force-ansi-depthwise")))]
            {
                esp_nn_set_depthwise_conv_scratch_buf_ansi(scratch_ptr);
                esp_nn_depthwise_conv_s8_ansi(
                    &input_dims, input, &filter_dims, params.weights.as_ptr(), bias_ptr,
                    &output_dims, output, &dw_params, &q_data,
                );
            }
        }
    }

    Ok(())
}

/// Whether ESP-NN's fast depthwise kernel handles this input correctly.
///
/// On ESP32-S3 the SIMD path requires a 16-byte aligned input; anything else
/// must go to the ANSI kernel. Other targets' primary kernels are
/// alignment-agnostic, so this is always `true`.
#[cfg(all(feature = "esp32s3", not(feature = "force-ansi-depthwise")))]
#[inline]
fn depthwise_can_use_simd(input: *const i8) -> bool {
    (input as usize).is_multiple_of(16)
}
#[cfg(not(all(feature = "esp32s3", not(feature = "force-ansi-depthwise"))))]
#[inline]
fn depthwise_can_use_simd(_input: *const i8) -> bool {
    true
}

#[cfg(all(feature = "esp32s3", feature = "trace-ops", debug_assertions))]
#[inline]
fn hits_esp32s3_mult1_3x3_padded_path(
    params: &DepthwiseConv2dParams<'_>,
    pad_w: i32,
    pad_h: i32,
) -> bool {
    params.depth_multiplier == 1
        && params.input_shape[3] % 16 == 0
        && params.weights_shape[1] == 3
        && params.weights_shape[2] == 3
        && pad_w == 1
        && pad_h == 1
}

fn depthwise_quant_buffers() -> (
    &'static mut [i32; MAX_CHANNELS],
    &'static mut [i32; MAX_CHANNELS],
) {
    unsafe {
        (
            &mut *core::ptr::addr_of_mut!(DEPTHWISE_MULT),
            &mut *core::ptr::addr_of_mut!(DEPTHWISE_SHIFT),
        )
    }
}

#[inline]
fn normalize_esp_nn_dilation(dilation: i32) -> i32 {
    dilation.saturating_sub(1)
}

pub fn scratch_size(
    input_shape: [usize; 4],
    weights_shape: [usize; 4],
    output_shape: [usize; 4],
) -> usize {
    if !shape4_fits_i32(input_shape)
        || !shape4_fits_i32(weights_shape)
        || !shape4_fits_i32(output_shape)
    {
        return 0;
    }

    let input_dims = io_dims_from_nhwc(input_shape);
    let filter_dims = depthwise_filter_dims(weights_shape);
    let output_dims = io_dims_from_nhwc(output_shape);
    let depth_multiplier = if input_shape[3] == 0 || !output_shape[3].is_multiple_of(input_shape[3])
    {
        return 0;
    } else {
        (output_shape[3] / input_shape[3]) as i32
    };
    let (stride_w, stride_h) = infer_stride(input_shape, weights_shape, output_shape);
    let pad_w = compute_padding(
        input_shape[2] as i32,
        output_shape[2] as i32,
        stride_w,
        weights_shape[2] as i32,
    );
    let pad_h = compute_padding(
        input_shape[1] as i32,
        output_shape[1] as i32,
        stride_h,
        weights_shape[1] as i32,
    );
    let dw_params = dw_conv_params_t {
        ch_mult: depth_multiplier,
        stride: data_2d_t {
            width: stride_w,
            height: stride_h,
        },
        padding: data_2d_t {
            width: pad_w,
            height: pad_h,
        },
        ..zero_dw_params()
    };
    let esp_size = unsafe {
        esp_nn_get_depthwise_conv_scratch_size(&input_dims, &filter_dims, &output_dims, &dw_params)
    }
    .max(0) as usize;

    // Reserve one alignment word so `run` can bump the scratch base to a 16-byte
    // boundary (ESP-NN's ESP32-S3 kernels assume an aligned scratch base; a
    // `Vec<u8>` only guarantees 1-byte alignment).
    if esp_size == 0 {
        0
    } else {
        esp_size + super::SCRATCH_ALIGN
    }
}

#[inline]
fn io_dims_from_nhwc(shape: [usize; 4]) -> data_dims_t {
    data_dims_t {
        width: shape[2] as i32,
        height: shape[1] as i32,
        channels: shape[3] as i32,
        extra: shape[0] as i32,
    }
}

#[inline]
fn depthwise_filter_dims(shape: [usize; 4]) -> data_dims_t {
    let _ = shape;
    data_dims_t {
        width: shape[2] as i32,
        height: shape[1] as i32,
        // Match the validated esp-nn kernel tests for depthwise filter dims.
        channels: 0,
        extra: 0,
    }
}

#[inline]
fn zero_dw_params() -> dw_conv_params_t {
    dw_conv_params_t {
        in_offset: 0,
        out_offset: 0,
        ch_mult: 0,
        stride: data_2d_t {
            width: 0,
            height: 0,
        },
        padding: data_2d_t {
            width: 0,
            height: 0,
        },
        dilation: data_2d_t {
            width: 0,
            height: 0,
        },
        activation: act_params_t { min: 0, max: 0 },
    }
}

#[inline]
fn shape4_fits_i32(shape: [usize; 4]) -> bool {
    shape.iter().all(|dim| *dim <= i32::MAX as usize)
}

#[inline]
fn tensor_len(shape: [usize; 4]) -> Option<usize> {
    shape
        .into_iter()
        .try_fold(1usize, |acc, dim| acc.checked_mul(dim))
}

#[inline]
fn infer_stride(
    input_shape: [usize; 4],
    weights_shape: [usize; 4],
    output_shape: [usize; 4],
) -> (i32, i32) {
    (
        infer_stride_dim(input_shape[2], weights_shape[2], output_shape[2]),
        infer_stride_dim(input_shape[1], weights_shape[1], output_shape[1]),
    )
}

#[inline]
fn infer_stride_dim(input: usize, filter: usize, output: usize) -> i32 {
    if output <= 1 {
        return 1;
    }

    let same_stride = input.div_ceil(output).max(1);
    let valid_stride = if input >= filter {
        (input - filter + 1).div_ceil(output).max(1)
    } else {
        same_stride
    };

    if ((output - 1) * valid_stride + filter) <= input {
        valid_stride as i32
    } else {
        same_stride as i32
    }
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

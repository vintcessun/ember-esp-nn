#[cfg(all(feature = "esp32s3", not(feature = "force-ansi-conv")))]
use esp_nn_sys::bindings::{
    esp_nn_conv_s8_esp32s3 as esp_nn_conv_s8,
    esp_nn_get_conv_scratch_size_esp32s3 as esp_nn_get_conv_scratch_size,
    esp_nn_set_conv_scratch_buf_esp32s3 as esp_nn_set_conv_scratch_buf,
};
#[cfg(all(feature = "esp32p4", not(feature = "force-ansi-conv")))]
use esp_nn_sys::bindings::{
    esp_nn_conv_s8_esp32p4 as esp_nn_conv_s8,
    esp_nn_get_conv_scratch_size_esp32p4 as esp_nn_get_conv_scratch_size,
    esp_nn_set_conv_scratch_buf_esp32p4 as esp_nn_set_conv_scratch_buf,
};
#[cfg(any(
    feature = "force-ansi-conv",
    not(any(feature = "esp32s3", feature = "esp32p4"))
))]
use esp_nn_sys::bindings::{
    esp_nn_conv_s8_ansi as esp_nn_conv_s8,
    esp_nn_get_conv_scratch_size_ansi as esp_nn_get_conv_scratch_size,
    esp_nn_set_conv_scratch_buf_ansi as esp_nn_set_conv_scratch_buf,
};
// ESP32-S3 only: correct, alignment-agnostic fallback. The optimized S3 conv
// (notably the 1x1/pointwise path) reads `input_data` with aligned SIMD loads
// and produces garbage when the input tensor is not 16-byte aligned. The
// `#[model]` macro's activation tensors are only 1-byte aligned, so when the
// input isn't aligned we run the reference ANSI kernel instead (see
// [`conv_can_use_simd`]).
#[cfg(all(feature = "esp32s3", not(feature = "force-ansi-conv")))]
use esp_nn_sys::bindings::{esp_nn_conv_s8_ansi, esp_nn_set_conv_scratch_buf_ansi};

use crate::quant::quantize_multiplier;
use ember_infer_core::{Conv2dParams, FusedActivation, KernelError, Padding, Status};
use esp_nn_sys::bindings::{act_params_t, conv_params_t, data_2d_t, data_dims_t, quant_data_t};

// Compile-time stack limit for per-channel quantization. Raise this if a model
// needs more output channels and the target stack budget allows it.
const MAX_CHANNELS: usize = 512;
static mut CONV_MULT: [i32; MAX_CHANNELS] = [0; MAX_CHANNELS];
static mut CONV_SHIFT: [i32; MAX_CHANNELS] = [0; MAX_CHANNELS];

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

pub fn run(params: Conv2dParams<'_>) -> Status {
    let c_out = params.weights_shape[0];
    if c_out == 0 || c_out > MAX_CHANNELS || c_out != params.output_shape[3] {
        return invalid_shape("conv2d invalid channels");
    }

    if !shape4_fits_i32(params.input_shape)
        || !shape4_fits_i32(params.weights_shape)
        || !shape4_fits_i32(params.output_shape)
        || tensor_len(params.input_shape).is_none_or(|len| params.input.len() < len)
        || tensor_len(params.weights_shape).is_none_or(|len| params.weights.len() < len)
        || tensor_len(params.output_shape).is_none_or(|len| params.output.len() < len)
        || params.bias.is_some_and(|bias| bias.len() < c_out)
    {
        return invalid_shape("conv2d invalid tensor shape");
    }

    let (mult_arr, shift_arr) = conv_quant_buffers();
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
    let filter_dims = conv_filter_dims(params.weights_shape);
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

    let c_params = conv_params_t {
        in_offset: -params.input_quant.zero_point,
        out_offset: params.output_quant.zero_point,
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

    let scratch_ptr = super::aligned_scratch_ptr(params.scratch);
    let input = params.input.as_ptr();
    let bias_ptr = params.bias.map_or(core::ptr::null(), |bias| bias.as_ptr());
    let output = params.output.as_mut_ptr();

    unsafe {
        if conv_can_use_simd(input) {
            esp_nn_set_conv_scratch_buf(scratch_ptr);
            esp_nn_conv_s8(
                &input_dims, input, &filter_dims, params.weights.as_ptr(), bias_ptr, &output_dims,
                output, &c_params, &q_data,
            );
        } else {
            // Only reachable on ESP32-S3 (elsewhere `conv_can_use_simd` is always
            // true and the primary kernel is alignment-agnostic).
            #[cfg(all(feature = "esp32s3", not(feature = "force-ansi-conv")))]
            {
                esp_nn_set_conv_scratch_buf_ansi(scratch_ptr);
                esp_nn_conv_s8_ansi(
                    &input_dims, input, &filter_dims, params.weights.as_ptr(), bias_ptr,
                    &output_dims, output, &c_params, &q_data,
                );
            }
        }
    }

    Ok(())
}

/// Whether ESP-NN's fast conv kernel handles this input correctly.
///
/// On ESP32-S3 the SIMD path requires a 16-byte aligned input; anything else
/// must go to the ANSI kernel. Other targets' primary kernels are
/// alignment-agnostic, so this is always `true`.
#[cfg(all(feature = "esp32s3", not(feature = "force-ansi-conv")))]
#[inline]
fn conv_can_use_simd(input: *const i8) -> bool {
    (input as usize).is_multiple_of(16)
}
#[cfg(not(all(feature = "esp32s3", not(feature = "force-ansi-conv"))))]
#[inline]
fn conv_can_use_simd(_input: *const i8) -> bool {
    true
}

fn conv_quant_buffers() -> (&'static mut [i32; MAX_CHANNELS], &'static mut [i32; MAX_CHANNELS]) {
    unsafe {
        (
            &mut *core::ptr::addr_of_mut!(CONV_MULT),
            &mut *core::ptr::addr_of_mut!(CONV_SHIFT),
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
    let filter_dims = conv_filter_dims(weights_shape);
    let output_dims = io_dims_from_nhwc(output_shape);
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
    let c_params = conv_params_t {
        stride: data_2d_t {
            width: stride_w,
            height: stride_h,
        },
        padding: data_2d_t {
            width: pad_w,
            height: pad_h,
        },
        ..zero_conv_params()
    };
    let size =
        unsafe { esp_nn_get_conv_scratch_size(&input_dims, &filter_dims, &output_dims, &c_params) };

    scratch_size_with_align(size)
}

/// Add slack so the scratch base can be bumped up to a 16-byte boundary at call
/// time (see [`super::aligned_scratch_ptr`]). ESP-NN requires a 16-byte aligned
/// scratch base; a bare `Vec<u8>` only guarantees 1-byte alignment.
#[inline]
fn scratch_size_with_align(size: i32) -> usize {
    let size = size.max(0) as usize;
    if size == 0 {
        0
    } else {
        size + super::SCRATCH_ALIGN
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
fn conv_filter_dims(shape: [usize; 4]) -> data_dims_t {
    data_dims_t {
        width: shape[2] as i32,
        height: shape[1] as i32,
        channels: shape[3] as i32,
        // Match the validated esp-nn kernel tests: conv filter extra is 1.
        extra: 1,
    }
}

#[inline]
fn zero_conv_params() -> conv_params_t {
    conv_params_t {
        in_offset: 0,
        out_offset: 0,
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

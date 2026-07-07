#[cfg(all(feature = "esp32s3", not(feature = "force-ansi-mul")))]
use esp_nn_sys::bindings::{
    esp_nn_mul_broadcast_channel_s8_esp32s3 as esp_nn_mul_broadcast_channel_s8,
    esp_nn_mul_elementwise_s8_esp32s3 as esp_nn_mul_elementwise_s8,
};
#[cfg(all(feature = "esp32p4", not(feature = "force-ansi-mul")))]
use esp_nn_sys::bindings::{
    esp_nn_mul_broadcast_channel_s8_esp32p4 as esp_nn_mul_broadcast_channel_s8,
    esp_nn_mul_elementwise_s8_esp32p4 as esp_nn_mul_elementwise_s8,
};
#[cfg(any(
    feature = "force-ansi-mul",
    not(any(feature = "esp32s3", feature = "esp32p4"))
))]
use esp_nn_sys::bindings::{
    esp_nn_mul_broadcast_channel_s8_ansi as esp_nn_mul_broadcast_channel_s8,
    esp_nn_mul_elementwise_s8_ansi as esp_nn_mul_elementwise_s8,
};
// ESP32-S3 only: alignment-agnostic fallbacks. The S3 mul kernels read the dense
// input with aligned SIMD loads, so unaligned inputs are read shifted (silently
// wrong).
#[cfg(all(feature = "esp32s3", not(feature = "force-ansi-mul")))]
use esp_nn_sys::bindings::{
    esp_nn_mul_broadcast_channel_s8_ansi, esp_nn_mul_elementwise_s8_ansi,
};

use crate::quant::quantize_multiplier;
use ember_infer_core::{FusedActivation, KernelError, MulParams, Status};

/// Whether the S3 mul kernel can be used for these inputs (both 16-byte
/// aligned). Always `true` on non-S3 targets (alignment-agnostic primary).
#[cfg(all(feature = "esp32s3", not(feature = "force-ansi-mul")))]
#[inline]
fn mul_can_use_simd(a: *const i8, b: *const i8) -> bool {
    (a as usize).is_multiple_of(16) && (b as usize).is_multiple_of(16)
}
#[cfg(not(all(feature = "esp32s3", not(feature = "force-ansi-mul"))))]
#[inline]
fn mul_can_use_simd(_a: *const i8, _b: *const i8) -> bool {
    true
}

pub fn run(params: MulParams<'_>) -> Status {
    // Element-wise when the two inputs have equal length; otherwise the shorter
    // one is a per-channel operand broadcast over the trailing NHWC dimension.
    // Multiply is commutative, so make the longer operand the dense one.
    let (dense, dense_q, per_ch, per_ch_q) = if params.input1.len() >= params.input2.len() {
        (
            params.input1,
            params.input1_quant,
            params.input2,
            params.input2_quant,
        )
    } else {
        (
            params.input2,
            params.input2_quant,
            params.input1,
            params.input1_quant,
        )
    };

    if per_ch.is_empty()
        || dense.len() % per_ch.len() != 0
        || params.output.len() != dense.len()
        || dense.len() > i32::MAX as usize
    {
        return Err(KernelError::InvalidShape);
    }

    // out_multiplier = input1_scale * input2_scale / output_scale (TFLite MUL).
    let real_multiplier = (params.input1_quant.scale as f64 * params.input2_quant.scale as f64)
        / params.output_quant.scale as f64;
    let (out_mult, out_shift) = quantize_multiplier(real_multiplier);
    let (activation_min, activation_max) = activation_range(params.activation, params.output_quant);

    let dense_ptr = dense.as_ptr();
    let per_ch_ptr = per_ch.as_ptr();
    let out = params.output.as_mut_ptr();
    let dense_off = -dense_q.zero_point;
    let per_ch_off = -per_ch_q.zero_point;
    let out_off = params.output_quant.zero_point;
    let channels = per_ch.len();
    let use_simd = mul_can_use_simd(dense_ptr, per_ch_ptr);

    unsafe {
        if channels == dense.len() {
            // Element-wise.
            let size = dense.len() as i32;
            if use_simd {
                esp_nn_mul_elementwise_s8(
                    dense_ptr, per_ch_ptr, dense_off, per_ch_off, out, out_off, out_mult, out_shift,
                    activation_min, activation_max, size,
                );
            } else {
                #[cfg(all(feature = "esp32s3", not(feature = "force-ansi-mul")))]
                esp_nn_mul_elementwise_s8_ansi(
                    dense_ptr, per_ch_ptr, dense_off, per_ch_off, out, out_off, out_mult, out_shift,
                    activation_min, activation_max, size,
                );
            }
        } else {
            // Per-channel broadcast: `per_ch` has `channels` elements, `dense`
            // has `total_spatial * channels`.
            let total_spatial = (dense.len() / channels) as i32;
            let channels = channels as i32;
            if use_simd {
                esp_nn_mul_broadcast_channel_s8(
                    dense_ptr, per_ch_ptr, dense_off, per_ch_off, out, out_off, out_mult, out_shift,
                    activation_min, activation_max, total_spatial, channels,
                );
            } else {
                #[cfg(all(feature = "esp32s3", not(feature = "force-ansi-mul")))]
                esp_nn_mul_broadcast_channel_s8_ansi(
                    dense_ptr, per_ch_ptr, dense_off, per_ch_off, out, out_off, out_mult, out_shift,
                    activation_min, activation_max, total_spatial, channels,
                );
            }
        }
    }

    Ok(())
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

#[cfg(all(feature = "esp32s3", not(feature = "force-ansi-add")))]
use esp_nn_sys::bindings::esp_nn_add_elementwise_s8_esp32s3 as esp_nn_add_elementwise_s8;
#[cfg(all(feature = "esp32p4", not(feature = "force-ansi-add")))]
use esp_nn_sys::bindings::esp_nn_add_elementwise_s8_esp32p4 as esp_nn_add_elementwise_s8;
#[cfg(any(
    feature = "force-ansi-add",
    not(any(feature = "esp32s3", feature = "esp32p4"))
))]
use esp_nn_sys::bindings::esp_nn_add_elementwise_s8_ansi as esp_nn_add_elementwise_s8;
// ESP32-S3 only: alignment-agnostic fallback. The S3 elementwise-add kernel
// reads both inputs with aligned SIMD loads, so unaligned inputs are read
// shifted (silently wrong).
#[cfg(all(feature = "esp32s3", not(feature = "force-ansi-add")))]
use esp_nn_sys::bindings::esp_nn_add_elementwise_s8_ansi;

use crate::quant::quantize_multiplier;
use ember_infer_core::{ElementwiseAddParams, FusedActivation, KernelError, Status};

const LEFT_SHIFT: i32 = 20;

/// Whether the S3 add kernel can be used for these inputs (both 16-byte
/// aligned). Always `true` on non-S3 targets (alignment-agnostic primary).
#[cfg(all(feature = "esp32s3", not(feature = "force-ansi-add")))]
#[inline]
fn add_can_use_simd(a: *const i8, b: *const i8) -> bool {
    (a as usize).is_multiple_of(16) && (b as usize).is_multiple_of(16)
}
#[cfg(not(all(feature = "esp32s3", not(feature = "force-ansi-add"))))]
#[inline]
fn add_can_use_simd(_a: *const i8, _b: *const i8) -> bool {
    true
}

pub fn run(params: ElementwiseAddParams<'_>) -> Status {
    if params.input1.len() > i32::MAX as usize
        || params.input2.len() < params.input1.len()
        || params.output.len() < params.input1.len()
    {
        return Err(KernelError::InvalidShape);
    }

    let input1_scale = params.input1_quant.scale as f64;
    let input2_scale = params.input2_quant.scale as f64;
    let output_scale = params.output_quant.scale as f64;

    let twice_max_input_scale = 2.0 * input1_scale.max(input2_scale);
    let real_input1_multiplier = input1_scale / twice_max_input_scale;
    let real_input2_multiplier = input2_scale / twice_max_input_scale;
    let real_output_multiplier =
        twice_max_input_scale / ((1_i64 << LEFT_SHIFT) as f64 * output_scale);

    let (input1_mult, input1_shift) = quantize_multiplier(real_input1_multiplier);
    let (input2_mult, input2_shift) = quantize_multiplier(real_input2_multiplier);
    let (out_mult, out_shift) = quantize_multiplier(real_output_multiplier);
    let (activation_min, activation_max) = activation_range(params.activation, params.output_quant);

    let in1 = params.input1.as_ptr();
    let in2 = params.input2.as_ptr();
    let out = params.output.as_mut_ptr();
    let zp1 = -params.input1_quant.zero_point;
    let zp2 = -params.input2_quant.zero_point;
    let out_zp = params.output_quant.zero_point;
    let len = params.input1.len() as i32;

    unsafe {
        if add_can_use_simd(in1, in2) {
            esp_nn_add_elementwise_s8(
                in1, in2, zp1, zp2, input1_mult, input2_mult, input1_shift, input2_shift,
                LEFT_SHIFT, out, out_zp, out_mult, out_shift, activation_min, activation_max, len,
            );
        } else {
            // Only reachable on ESP32-S3 (elsewhere `add_can_use_simd` is always
            // true and the primary kernel is alignment-agnostic).
            #[cfg(all(feature = "esp32s3", not(feature = "force-ansi-add")))]
            esp_nn_add_elementwise_s8_ansi(
                in1, in2, zp1, zp2, input1_mult, input2_mult, input1_shift, input2_shift,
                LEFT_SHIFT, out, out_zp, out_mult, out_shift, activation_min, activation_max, len,
            );
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

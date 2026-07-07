#[cfg(all(feature = "esp32s3", not(feature = "force-ansi-softmax")))]
use esp_nn_sys::bindings::{
    esp_nn_get_softmax_scratch_size_esp32s3 as esp_nn_get_softmax_scratch_size,
    esp_nn_set_softmax_scratch_buf_esp32s3 as esp_nn_set_softmax_scratch_buf,
    esp_nn_softmax_s8_esp32s3 as esp_nn_softmax_s8,
};
#[cfg(all(feature = "esp32p4", not(feature = "force-ansi-softmax")))]
use esp_nn_sys::bindings::{
    esp_nn_get_softmax_scratch_size_esp32p4 as esp_nn_get_softmax_scratch_size,
    esp_nn_set_softmax_scratch_buf_esp32p4 as esp_nn_set_softmax_scratch_buf,
    esp_nn_softmax_s8_esp32p4 as esp_nn_softmax_s8,
};
#[cfg(any(
    feature = "force-ansi-softmax",
    not(any(feature = "esp32s3", feature = "esp32p4"))
))]
use esp_nn_sys::bindings::{
    esp_nn_get_softmax_scratch_size_ansi as esp_nn_get_softmax_scratch_size,
    esp_nn_set_softmax_scratch_buf_ansi as esp_nn_set_softmax_scratch_buf,
    esp_nn_softmax_s8_ansi as esp_nn_softmax_s8,
};
// ESP32-S3 only: alignment-agnostic fallback. The S3 softmax uses an aligned
// SIMD max-reduction over the input for rows of length >= 32, so an unaligned
// input is read shifted (silently wrong) for larger class counts.
#[cfg(all(feature = "esp32s3", not(feature = "force-ansi-softmax")))]
use esp_nn_sys::bindings::{
    esp_nn_set_softmax_scratch_buf_ansi, esp_nn_softmax_s8_ansi,
};

use crate::quant::quantize_multiplier;
use ember_infer_core::{KernelError, SoftmaxParams, Status};

const INPUT_INTEGER_BITS: i32 = 5;

/// Whether the S3 softmax can be used for this input (16-byte aligned). Always
/// `true` on non-S3 targets (alignment-agnostic primary).
#[cfg(all(feature = "esp32s3", not(feature = "force-ansi-softmax")))]
#[inline]
fn softmax_can_use_simd(input: *const i8) -> bool {
    (input as usize).is_multiple_of(16)
}
#[cfg(not(all(feature = "esp32s3", not(feature = "force-ansi-softmax"))))]
#[inline]
fn softmax_can_use_simd(_input: *const i8) -> bool {
    true
}

pub fn run(params: SoftmaxParams<'_>) -> Status {
    let batch = params.input_shape[0];
    let num_classes = params.input_shape[1];

    if batch > i32::MAX as usize
        || num_classes > i32::MAX as usize
        || batch
            .checked_mul(num_classes)
            .is_none_or(|len| params.input.len() < len || params.output.len() < len)
    {
        return Err(KernelError::InvalidShape);
    }

    let scale = params.beta as f64
        * params.input_quant.scale as f64
        * ((1_i64 << (31 - INPUT_INTEGER_BITS)) as f64);
    let (mult, shift) = quantize_multiplier(scale);
    let diff_min = -calculate_input_radius(INPUT_INTEGER_BITS, shift);

    let scratch_ptr = super::aligned_scratch_ptr(params.scratch);
    let input = params.input.as_ptr();
    let output = params.output.as_mut_ptr();

    unsafe {
        if softmax_can_use_simd(input) {
            esp_nn_set_softmax_scratch_buf(scratch_ptr);
            esp_nn_softmax_s8(input, batch as i32, num_classes as i32, mult, shift, diff_min, output);
        } else {
            // Only reachable on ESP32-S3 (elsewhere `softmax_can_use_simd` is
            // always true and the primary kernel is alignment-agnostic).
            #[cfg(all(feature = "esp32s3", not(feature = "force-ansi-softmax")))]
            {
                esp_nn_set_softmax_scratch_buf_ansi(scratch_ptr);
                esp_nn_softmax_s8_ansi(
                    input, batch as i32, num_classes as i32, mult, shift, diff_min, output,
                );
            }
        }
    }

    Ok(())
}

pub fn scratch_size(num_classes: usize) -> usize {
    if num_classes > i32::MAX as usize {
        return 0;
    }

    let size = unsafe { esp_nn_get_softmax_scratch_size(num_classes as i32, 1) };
    let size = size.max(0) as usize;
    // Reserve slack so the base can be bumped to a 16-byte boundary at call time
    // (see `super::aligned_scratch_ptr`).
    if size == 0 {
        0
    } else {
        size + super::SCRATCH_ALIGN
    }
}

#[inline]
fn calculate_input_radius(input_integer_bits: i32, input_left_shift: i32) -> i32 {
    let total_signed_bits = 31;
    let max_input_rescaled =
        ((1_i64 << input_integer_bits) - 1) * (1_i64 << (total_signed_bits - input_integer_bits));

    if input_left_shift <= 0 {
        max_input_rescaled as i32
    } else {
        (max_input_rescaled >> input_left_shift) as i32
    }
}

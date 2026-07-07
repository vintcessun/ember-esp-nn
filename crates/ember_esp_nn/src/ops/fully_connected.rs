#[cfg(any(
    feature = "force-ansi-fc",
    not(any(feature = "esp32s3", feature = "esp32p4"))
))]
use esp_nn_sys::bindings::{
    esp_nn_fully_connected_per_ch_s8_ansi as esp_nn_fully_connected_per_ch_s8,
    esp_nn_fully_connected_s8_ansi as esp_nn_fully_connected_s8,
};
#[cfg(all(feature = "esp32p4", not(feature = "force-ansi-fc")))]
use esp_nn_sys::bindings::{
    esp_nn_fully_connected_per_ch_s8_esp32p4 as esp_nn_fully_connected_per_ch_s8,
    esp_nn_fully_connected_s8_esp32p4 as esp_nn_fully_connected_s8,
};
#[cfg(all(feature = "esp32s3", not(feature = "force-ansi-fc")))]
use esp_nn_sys::bindings::{
    esp_nn_fully_connected_per_ch_s8_esp32s3 as esp_nn_fully_connected_per_ch_s8,
    esp_nn_fully_connected_s8_esp32s3 as esp_nn_fully_connected_s8,
};
// ESP32-S3 only: correct, alignment-agnostic fallbacks. The optimized S3 C entry
// (`esp_nn_fully_connected_*_s8_esp32s3`) silently drops to a hand-written s16
// assembly kernel whenever it can't take the SIMD path — i.e. when `input_data`
// is not 16-byte aligned, `row_len < 16`, or `filter_offset != 0` — and that s16
// fallback produces wrong results. The `#[model]` macro allocates activation
// tensors as 1-byte-aligned `[i8; N]` arrays, so the input alignment (and hence
// whether the buggy path is taken) depends on the stack layout and varies by
// build profile. We detect the same conditions in [`fc_can_use_simd`] and route
// those cases to the reference ANSI kernel instead, keeping the fast SIMD path
// only for inputs it handles correctly.
#[cfg(all(feature = "esp32s3", not(feature = "force-ansi-fc")))]
use esp_nn_sys::bindings::{
    esp_nn_fully_connected_per_ch_s8_ansi, esp_nn_fully_connected_s8_ansi,
};

use crate::quant::quantize_multiplier;
use ember_infer_core::{FullyConnectedParams, FusedActivation, KernelError, Status};

/// Whether the target's *primary* (fast) FC kernel handles this call correctly.
///
/// On ESP32-S3 the SIMD entry only handles 16-byte aligned inputs with
/// `row_len >= 16` and `filter_offset == 0`; anything else must go to the ANSI
/// kernel (see the import comment above). On other targets the primary kernel is
/// already alignment-agnostic, so this is always `true`.
#[cfg(all(feature = "esp32s3", not(feature = "force-ansi-fc")))]
#[inline]
fn fc_can_use_simd(filter_offset: i32, row_len: usize, input: *const i8) -> bool {
    filter_offset == 0 && row_len >= 16 && (input as usize).is_multiple_of(16)
}
#[cfg(not(all(feature = "esp32s3", not(feature = "force-ansi-fc"))))]
#[inline]
fn fc_can_use_simd(_filter_offset: i32, _row_len: usize, _input: *const i8) -> bool {
    true
}

/// Dispatch a per-channel FC call to the fast primary kernel, or the correct
/// ANSI fallback when the primary can't handle it (ESP32-S3 only).
#[allow(clippy::too_many_arguments)]
#[inline]
unsafe fn dispatch_fc_per_ch(
    use_simd: bool,
    input: *const i8,
    input_offset: i32,
    row_len: u16,
    weights: *const i8,
    weights_offset: i32,
    bias: *const i32,
    output: *mut i8,
    out_channels: u16,
    out_offset: i32,
    shift: *const i32,
    mult: *const i32,
    act_min: i32,
    act_max: i32,
) {
    #[cfg(all(feature = "esp32s3", not(feature = "force-ansi-fc")))]
    if !use_simd {
        unsafe {
            esp_nn_fully_connected_per_ch_s8_ansi(
                input, input_offset, row_len, weights, weights_offset, bias, output, out_channels,
                out_offset, shift, mult, act_min, act_max,
            );
        }
        return;
    }
    let _ = use_simd;
    unsafe {
        esp_nn_fully_connected_per_ch_s8(
            input, input_offset, row_len, weights, weights_offset, bias, output, out_channels,
            out_offset, shift, mult, act_min, act_max,
        );
    }
}

/// Dispatch a per-tensor FC call to the fast primary kernel, or the correct ANSI
/// fallback when the primary can't handle it (ESP32-S3 only).
#[allow(clippy::too_many_arguments)]
#[inline]
unsafe fn dispatch_fc(
    use_simd: bool,
    input: *const i8,
    input_offset: i32,
    row_len: u16,
    weights: *const i8,
    weights_offset: i32,
    bias: *const i32,
    output: *mut i8,
    out_channels: u16,
    out_offset: i32,
    shift: i32,
    mult: i32,
    act_min: i32,
    act_max: i32,
) {
    #[cfg(all(feature = "esp32s3", not(feature = "force-ansi-fc")))]
    if !use_simd {
        unsafe {
            esp_nn_fully_connected_s8_ansi(
                input, input_offset, row_len, weights, weights_offset, bias, output, out_channels,
                out_offset, shift, mult, act_min, act_max,
            );
        }
        return;
    }
    let _ = use_simd;
    unsafe {
        esp_nn_fully_connected_s8(
            input, input_offset, row_len, weights, weights_offset, bias, output, out_channels,
            out_offset, shift, mult, act_min, act_max,
        );
    }
}

const MAX_FC_PER_CHANNEL_PARAMS: usize = 1024;
static mut FC_MULT: [i32; MAX_FC_PER_CHANNEL_PARAMS] = [0; MAX_FC_PER_CHANNEL_PARAMS];
static mut FC_SHIFT: [i32; MAX_FC_PER_CHANNEL_PARAMS] = [0; MAX_FC_PER_CHANNEL_PARAMS];

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

pub fn run(params: FullyConnectedParams<'_>) -> Status {
    let row_len = params.weights_shape[1];
    let out_channels = params.output_depth;

    if row_len > u16::MAX as usize || out_channels > u16::MAX as usize {
        return invalid_shape("fully_connected invalid dims");
    }

    let Some(weights_len) = row_len.checked_mul(out_channels) else {
        return invalid_shape("fully_connected weights_len overflow");
    };

    if params.input.len() < row_len
        || params.weights.len() < weights_len
        || params.output.len() < out_channels
        || params.bias.is_some_and(|bias| bias.len() < out_channels)
    {
        return invalid_shape("fully_connected invalid tensor shape");
    }

    let (activation_min, activation_max) = activation_range(params.activation, params.output_quant);

    if let Some(per_channel) = params.weights_per_channel_quant {
        if out_channels > MAX_FC_PER_CHANNEL_PARAMS {
            return invalid_shape("fully_connected per-channel overflow");
        }

        let (mult_arr, shift_arr) = fc_quant_buffers();

        for i in 0..out_channels {
            let weight_scale = per_channel
                .scales
                .get(i)
                .copied()
                .unwrap_or(params.weights_quant.scale);
            let effective_scale =
                (params.input_quant.scale * weight_scale) / params.output_quant.scale;
            let (out_mult, out_shift) = quantize_multiplier(effective_scale as f64);
            mult_arr[i] = out_mult;
            shift_arr[i] = out_shift;
        }

        let filter_offset = -params.weights_quant.zero_point;
        let use_simd = fc_can_use_simd(filter_offset, row_len, params.input.as_ptr());
        unsafe {
            dispatch_fc_per_ch(
                use_simd,
                params.input.as_ptr(),
                -params.input_quant.zero_point,
                row_len as u16,
                params.weights.as_ptr(),
                filter_offset,
                params.bias.map_or(core::ptr::null(), |bias| bias.as_ptr()),
                params.output.as_mut_ptr(),
                out_channels as u16,
                params.output_quant.zero_point,
                shift_arr[..out_channels].as_ptr(),
                mult_arr[..out_channels].as_ptr(),
                activation_min,
                activation_max,
            );
        }
    } else {
        let effective_scale =
            (params.input_quant.scale * params.weights_quant.scale) / params.output_quant.scale;
        let (out_mult, out_shift) = quantize_multiplier(effective_scale as f64);

        let filter_offset = -params.weights_quant.zero_point;
        let use_simd = fc_can_use_simd(filter_offset, row_len, params.input.as_ptr());
        unsafe {
            dispatch_fc(
                use_simd,
                params.input.as_ptr(),
                -params.input_quant.zero_point,
                row_len as u16,
                params.weights.as_ptr(),
                filter_offset,
                params.bias.map_or(core::ptr::null(), |bias| bias.as_ptr()),
                params.output.as_mut_ptr(),
                out_channels as u16,
                params.output_quant.zero_point,
                out_shift,
                out_mult,
                activation_min,
                activation_max,
            );
        }
    }

    Ok(())
}

fn fc_quant_buffers() -> (
    &'static mut [i32; MAX_FC_PER_CHANNEL_PARAMS],
    &'static mut [i32; MAX_FC_PER_CHANNEL_PARAMS],
) {
    unsafe {
        (
            &mut *core::ptr::addr_of_mut!(FC_MULT),
            &mut *core::ptr::addr_of_mut!(FC_SHIFT),
        )
    }
}

#[inline]
fn activation_range(
    act: FusedActivation,
    output_quant: ember_infer_core::QuantParam,
) -> (i32, i32) {
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

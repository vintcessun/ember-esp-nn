#[cfg(not(any(feature = "esp32s3", feature = "esp32p4")))]
use esp_nn_sys::bindings::esp_nn_fully_connected_s8_ansi as esp_nn_fully_connected_s8;
#[cfg(any(feature = "esp32s3", feature = "esp32p4"))]
use esp_nn_sys::bindings::esp_nn_fully_connected_s8_esp32s3 as esp_nn_fully_connected_s8;

use crate::quant::quantize_multiplier;
use ember_infer_core::{FullyConnectedParams, FusedActivation, KernelError, Status};

pub fn run(params: FullyConnectedParams<'_>) -> Status {
    let row_len = params.weights_shape[1];
    let out_channels = params.output_depth;

    if row_len > u16::MAX as usize || out_channels > u16::MAX as usize {
        return Err(KernelError::InvalidShape);
    }

    let Some(weights_len) = row_len.checked_mul(out_channels) else {
        return Err(KernelError::InvalidShape);
    };

    if params.input.len() < row_len
        || params.weights.len() < weights_len
        || params.output.len() < out_channels
        || params.bias.is_some_and(|bias| bias.len() < out_channels)
    {
        return Err(KernelError::InvalidShape);
    }

    let effective_scale =
        (params.input_quant.scale * params.weights_quant.scale) / params.output_quant.scale;
    let (out_mult, out_shift) = quantize_multiplier(effective_scale as f64);
    let (activation_min, activation_max) =
        activation_range(params.activation, params.output_quant.zero_point);

    unsafe {
        esp_nn_fully_connected_s8(
            params.input.as_ptr(),
            -params.input_quant.zero_point,
            row_len as u16,
            params.weights.as_ptr(),
            -params.weights_quant.zero_point,
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

    Ok(())
}

#[inline]
fn activation_range(act: FusedActivation, out_zero_point: i32) -> (i32, i32) {
    match act {
        FusedActivation::None => (-128, 127),
        FusedActivation::Relu | FusedActivation::Relu6 => (out_zero_point.max(-128), 127),
        _ => (-128, 127),
    }
}

//! Standalone hardware test for the ESP-NN `MUL` operator wiring
//! (`EspBackend::mul`), exercising both the element-wise and per-channel
//! broadcast paths and cross-checking against a float reference.
//!
//! No `.tflite` model in `models/` currently uses `MUL`, so this verifies the
//! new operator directly.
#![no_std]
#![no_main]

use defmt::info;
use ember_esp_nn::EspBackend;
use ember_infer_core::{FusedActivation, KernelBackend, MulParams, QuantParam};
use esp_hal::clock::CpuClock;
use esp_hal::main;
use panic_rtt_target as _;

esp_bootloader_esp_idf::esp_app_desc!();

// 16-byte aligned storage so the ESP32-S3 SIMD path is exercised (not the ANSI
// fallback).
#[repr(C, align(16))]
struct A16<const N: usize>([i8; N]);

const IN1_SCALE: f32 = 0.018;
const IN1_ZP: i32 = -3;
const IN2_SCALE: f32 = 0.025;
const IN2_ZP: i32 = 7;
const OUT_SCALE: f32 = 0.04;
const OUT_ZP: i32 = -10;

fn q1() -> QuantParam {
    QuantParam {
        scale: IN1_SCALE,
        zero_point: IN1_ZP,
    }
}
fn q2() -> QuantParam {
    QuantParam {
        scale: IN2_SCALE,
        zero_point: IN2_ZP,
    }
}
fn qo() -> QuantParam {
    QuantParam {
        scale: OUT_SCALE,
        zero_point: OUT_ZP,
    }
}

/// Float reference matching `ember-infer-ref`'s `mul`.
fn ref_mul(a: i8, b: i8) -> i8 {
    let lhs = (a as i32 - IN1_ZP) as f32 * IN1_SCALE;
    let rhs = (b as i32 - IN2_ZP) as f32 * IN2_SCALE;
    let q = libm::roundf((lhs * rhs) / OUT_SCALE) as i32 + OUT_ZP;
    q.clamp(-128, 127) as i8
}

fn check(name: &str, got: &[i8], expect_ab: impl Fn(usize) -> i8) -> bool {
    let mut max_err = 0i32;
    let mut fails = 0u32;
    for (i, &g) in got.iter().enumerate() {
        let e = expect_ab(i);
        let err = (g as i32 - e as i32).abs();
        max_err = max_err.max(err);
        // ESP-NN uses fixed-point requant vs the float reference; allow ±1 LSB.
        if err > 1 {
            fails += 1;
        }
    }
    let ok = fails == 0;
    info!(
        "[mul] {} len={} max_err={} fails={} {}",
        name,
        got.len(),
        max_err,
        fails,
        if ok { "PASS" } else { "FAIL" }
    );
    ok
}

#[main]
fn main() -> ! {
    rtt_target::rtt_init_defmt!();
    let config = esp_hal::Config::default().with_cpu_clock(CpuClock::max());
    let _peripherals = esp_hal::init(config);

    info!("=== ember-esp-nn MUL operator test ===");
    let mut backend = EspBackend;
    let mut all_ok = true;

    // ---- Element-wise: 64 values covering the int8 range ----
    {
        const N: usize = 64;
        let mut in1 = A16([0i8; N]);
        let mut in2 = A16([0i8; N]);
        for i in 0..N {
            in1.0[i] = ((i as i32) * 4 - 128) as i8;
            in2.0[i] = (120 - (i as i32) * 3) as i8;
        }
        let mut out = A16([0i8; N]);
        backend
            .mul(MulParams {
                input1: &in1.0,
                input1_quant: q1(),
                input2: &in2.0,
                input2_quant: q2(),
                output: &mut out.0,
                output_quant: qo(),
                activation: FusedActivation::None,
            })
            .expect("mul elementwise failed");
        all_ok &= check("elementwise", &out.0, |i| ref_mul(in1.0[i], in2.0[i]));
    }

    // ---- Per-channel broadcast: dense [spatial=5, C=16], per-ch [16] ----
    {
        const C: usize = 16;
        const SP: usize = 5;
        const N: usize = SP * C;
        let mut dense = A16([0i8; N]);
        let mut perch = A16([0i8; C]);
        for i in 0..N {
            dense.0[i] = ((i as i32 * 7) % 255 - 128) as i8;
        }
        for c in 0..C {
            perch.0[c] = ((c as i32) * 15 - 100) as i8;
        }
        let mut out = A16([0i8; N]);
        backend
            .mul(MulParams {
                input1: &dense.0,
                input1_quant: q1(),
                input2: &perch.0,
                input2_quant: q2(),
                output: &mut out.0,
                output_quant: qo(),
                activation: FusedActivation::None,
            })
            .expect("mul broadcast failed");
        all_ok &= check("broadcast", &out.0, |i| ref_mul(dense.0[i], perch.0[i % C]));
    }

    if all_ok {
        info!("=== MUL TEST PASSED ===");
    } else {
        panic!("MUL test failed");
    }

    loop {
        core::hint::spin_loop();
    }
}

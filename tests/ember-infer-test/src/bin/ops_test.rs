//! Backend-level correctness test for the `EspBackend` operator wrappers that no
//! `.tflite` model in `models/` exercises end-to-end: `mul`, `add`, `avg_pool`,
//! `max_pool`.
//!
//! For each op we run it twice — once with a 16-byte aligned input (ESP32-S3
//! SIMD path) and once with a deliberately mis-aligned input (ANSI fallback
//! path) — and check that:
//!   * both paths agree (validates the alignment dispatch), and
//!   * both match an independent reference (validates the wrapper's param math:
//!     quantization multiplier/shift, padding, fused activation).
#![no_std]
#![no_main]

use defmt::info;
use ember_esp_nn::EspBackend;
use ember_infer_core::{
    ElementwiseAddParams, FusedActivation, KernelBackend, MulParams, Padding, PoolParams,
    QuantParam,
};
use esp_hal::clock::CpuClock;
use esp_hal::main;
use panic_rtt_target as _;

esp_bootloader_esp_idf::esp_app_desc!();

/// 16-byte aligned backing buffer. `&.0[0..n]` is 16-aligned; `&.0[1..1+n]` is
/// 1-aligned (forces the ANSI path).
#[repr(C, align(16))]
struct Al<const N: usize>([i8; N]);

const Q1: QuantParam = QuantParam {
    scale: 0.018,
    zero_point: -3,
};
const Q2: QuantParam = QuantParam {
    scale: 0.025,
    zero_point: 7,
};
const QO: QuantParam = QuantParam {
    scale: 0.04,
    zero_point: -10,
};
// Pooling uses equal in/out quant so the reference is exact.
const QP: QuantParam = QuantParam {
    scale: 0.05,
    zero_point: -6,
};

fn dq(v: i8, q: QuantParam) -> f32 {
    (v as i32 - q.zero_point) as f32 * q.scale
}
fn requant(x: f32, q: QuantParam) -> i8 {
    (libm::roundf(x / q.scale) as i32 + q.zero_point).clamp(-128, 127) as i8
}

/// Compare two slices; returns (max_abs_err, count_exceeding_tol).
fn diff(a: &[i8], b: &[i8], tol: i32) -> (i32, u32) {
    let mut max = 0;
    let mut fails = 0;
    for (x, y) in a.iter().zip(b.iter()) {
        let e = (*x as i32 - *y as i32).abs();
        max = max.max(e);
        if e > tol {
            fails += 1;
        }
    }
    (max, fails)
}

fn report(name: &str, s3: &[i8], ansi: &[i8], reference: &[i8]) -> bool {
    let (d_disp, f_disp) = diff(s3, ansi, 1);
    let (d_s3, f_s3) = diff(s3, reference, 1);
    let (d_ansi, f_ansi) = diff(ansi, reference, 1);
    let ok = f_disp == 0 && f_s3 == 0 && f_ansi == 0;
    info!(
        "[ops] {} len={} s3-vs-ansi(err={} f={}) s3-vs-ref(err={} f={}) ansi-vs-ref(err={} f={}) {}",
        name,
        s3.len(),
        d_disp,
        f_disp,
        d_s3,
        f_s3,
        d_ansi,
        f_ansi,
        if ok { "PASS" } else { "FAIL" }
    );
    ok
}

fn test_mul(backend: &mut EspBackend) -> bool {
    const N: usize = 48;
    let mut src1 = [0i8; N];
    let mut src2 = [0i8; N];
    for i in 0..N {
        src1[i] = ((i as i32) * 5 - 120) as i8;
        src2[i] = (100 - (i as i32) * 4) as i8;
    }
    let mut reference = [0i8; N];
    for i in 0..N {
        reference[i] = requant(dq(src1[i], Q1) * dq(src2[i], Q2), QO);
    }
    let run = |backend: &mut EspBackend, off: usize, out: &mut [i8]| {
        let mut b1 = Al([0i8; N + 16]);
        let mut b2 = Al([0i8; N + 16]);
        b1.0[off..off + N].copy_from_slice(&src1);
        b2.0[off..off + N].copy_from_slice(&src2);
        backend
            .mul(MulParams {
                input1: &b1.0[off..off + N],
                input1_quant: Q1,
                input2: &b2.0[off..off + N],
                input2_quant: Q2,
                output: out,
                output_quant: QO,
                activation: FusedActivation::None,
            })
            .unwrap();
    };
    let mut s3 = [0i8; N];
    let mut ansi = [0i8; N];
    run(backend, 0, &mut s3);
    run(backend, 1, &mut ansi);
    report("mul", &s3, &ansi, &reference)
}

fn test_add(backend: &mut EspBackend) -> bool {
    const N: usize = 48;
    let mut src1 = [0i8; N];
    let mut src2 = [0i8; N];
    for i in 0..N {
        src1[i] = ((i as i32) * 5 - 120) as i8;
        src2[i] = (100 - (i as i32) * 4) as i8;
    }
    let mut reference = [0i8; N];
    for i in 0..N {
        reference[i] = requant(dq(src1[i], Q1) + dq(src2[i], Q2), QO);
    }
    let run = |backend: &mut EspBackend, off: usize, out: &mut [i8]| {
        let mut b1 = Al([0i8; N + 16]);
        let mut b2 = Al([0i8; N + 16]);
        b1.0[off..off + N].copy_from_slice(&src1);
        b2.0[off..off + N].copy_from_slice(&src2);
        backend
            .add(ElementwiseAddParams {
                input1: &b1.0[off..off + N],
                input1_quant: Q1,
                input2: &b2.0[off..off + N],
                input2_quant: Q2,
                output: out,
                output_quant: QO,
                activation: FusedActivation::None,
            })
            .unwrap();
    };
    let mut s3 = [0i8; N];
    let mut ansi = [0i8; N];
    run(backend, 0, &mut s3);
    run(backend, 1, &mut ansi);
    // ADD's fixed-point requant can differ from the float ref by up to ~2 LSB.
    let (d_disp, f_disp) = diff(&s3, &ansi, 1);
    let (d_ref, _) = diff(&s3, &reference, 2);
    let (_, f_ref) = diff(&s3, &reference, 2);
    let ok = f_disp == 0 && f_ref == 0;
    info!(
        "[ops] add len={} s3-vs-ansi(err={} f={}) s3-vs-ref(err={} tol=2) {}",
        N,
        d_disp,
        f_disp,
        d_ref,
        if ok { "PASS" } else { "FAIL" }
    );
    ok
}

// Pool: input [1,4,4,C], filter 2x2 stride 2, Valid -> output [1,2,2,C].
const PC: usize = 8;
const PIN: usize = 4 * 4 * PC; // 128
const POUT: usize = 2 * 2 * PC; // 32

fn pool_src() -> [i8; PIN] {
    let mut s = [0i8; PIN];
    for (i, v) in s.iter_mut().enumerate() {
        *v = (((i as i32) * 13) % 200 - 100) as i8;
    }
    s
}

fn test_max_pool(backend: &mut EspBackend) -> bool {
    let src = pool_src();
    // Reference: max over each 2x2 window per channel (equal in/out quant).
    let mut reference = [0i8; POUT];
    for oy in 0..2 {
        for ox in 0..2 {
            for c in 0..PC {
                let mut m = i8::MIN;
                for fy in 0..2 {
                    for fx in 0..2 {
                        let iy = oy * 2 + fy;
                        let ix = ox * 2 + fx;
                        m = m.max(src[(iy * 4 + ix) * PC + c]);
                    }
                }
                reference[(oy * 2 + ox) * PC + c] = m;
            }
        }
    }
    let run = |backend: &mut EspBackend, off: usize, out: &mut [i8]| {
        let mut b = Al([0i8; PIN + 16]);
        b.0[off..off + PIN].copy_from_slice(&src);
        backend
            .max_pool(PoolParams {
                input: &b.0[off..off + PIN],
                input_shape: [1, 4, 4, PC],
                input_quant: QP,
                output: out,
                output_shape: [1, 2, 2, PC],
                output_quant: QP,
                stride_w: 2,
                stride_h: 2,
                filter_w: 2,
                filter_h: 2,
                padding: Padding::Valid,
                activation: FusedActivation::None,
            })
            .unwrap();
    };
    let mut s3 = [0i8; POUT];
    let mut ansi = [0i8; POUT];
    run(backend, 0, &mut s3);
    run(backend, 1, &mut ansi);
    report("max_pool", &s3, &ansi, &reference)
}

fn test_avg_pool(backend: &mut EspBackend) -> bool {
    let src = pool_src();
    // Reference: round(sum(in - zp)/4) + zp (equal in/out quant, count=4).
    let mut reference = [0i8; POUT];
    for oy in 0..2 {
        for ox in 0..2 {
            for c in 0..PC {
                let mut acc = 0i32;
                for fy in 0..2 {
                    for fx in 0..2 {
                        let iy = oy * 2 + fy;
                        let ix = ox * 2 + fx;
                        acc += src[(iy * 4 + ix) * PC + c] as i32 - QP.zero_point;
                    }
                }
                let v = libm::roundf(acc as f32 / 4.0) as i32 + QP.zero_point;
                reference[(oy * 2 + ox) * PC + c] = v.clamp(-128, 127) as i8;
            }
        }
    }
    let run = |backend: &mut EspBackend, off: usize, out: &mut [i8]| {
        let mut b = Al([0i8; PIN + 16]);
        b.0[off..off + PIN].copy_from_slice(&src);
        backend
            .avg_pool(PoolParams {
                input: &b.0[off..off + PIN],
                input_shape: [1, 4, 4, PC],
                input_quant: QP,
                output: out,
                output_shape: [1, 2, 2, PC],
                output_quant: QP,
                stride_w: 2,
                stride_h: 2,
                filter_w: 2,
                filter_h: 2,
                padding: Padding::Valid,
                activation: FusedActivation::None,
            })
            .unwrap();
    };
    let mut s3 = [0i8; POUT];
    let mut ansi = [0i8; POUT];
    run(backend, 0, &mut s3);
    run(backend, 1, &mut ansi);
    report("avg_pool", &s3, &ansi, &reference)
}

#[main]
fn main() -> ! {
    rtt_target::rtt_init_defmt!();
    let config = esp_hal::Config::default().with_cpu_clock(CpuClock::max());
    let _peripherals = esp_hal::init(config);

    info!("=== ember-esp-nn backend ops test (aligned S3 vs unaligned ANSI vs ref) ===");
    let mut backend = EspBackend;
    let mut ok = true;
    ok &= test_mul(&mut backend);
    ok &= test_add(&mut backend);
    ok &= test_max_pool(&mut backend);
    ok &= test_avg_pool(&mut backend);

    if ok {
        info!("=== OPS TEST PASSED ===");
    } else {
        panic!("ops test failed");
    }

    loop {
        core::hint::spin_loop();
    }
}

#![no_std]
#![no_main]
#![deny(
    clippy::mem_forget,
    reason = "mem::forget is generally not safe to do with esp_hal types, especially those \
    holding buffers for the duration of a data transfer."
)]
#![deny(clippy::large_stack_frames)]

use defmt::info;
use embassy_executor::Spawner;
use embassy_time::{Duration, Instant, Timer};
use esp_hal::clock::CpuClock;
use esp_hal::timer::timg::TimerGroup;
use panic_rtt_target as _;

// This creates a default app-descriptor required by the esp-idf bootloader.
// For more information see: <https://docs.espressif.com/projects/esp-idf/en/stable/esp32/api-reference/system/app_image_format.html#application-description>
esp_bootloader_esp_idf::esp_app_desc!();

use ember_esp_nn::EspBackend;

const SINE_BENCH_ITERS: usize = 100_000;

#[allow(
    clippy::large_stack_frames,
    reason = "the model macro generates static model metadata for the hardware test"
)]
mod models {
    use ember_infer_macros::model;

    #[model("models/sine.tflite")]
    pub struct SineModel;
}

fn quantize(val: f32, scale: f32, zero_point: i32) -> i8 {
    let scaled = (val / scale) + zero_point as f32;
    let rounded = if scaled >= 0.0 {
        (scaled + 0.5) as i32
    } else {
        (scaled - 0.5) as i32
    };

    rounded.clamp(i8::MIN as i32, i8::MAX as i32) as i8
}

fn dequantize(val: i8, scale: f32, zero_point: i32) -> f32 {
    (val as i32 - zero_point) as f32 * scale
}

fn test_sine() {
    let cases: &[(f32, f32)] = &[
        (0.0, 0.0),
        (core::f32::consts::FRAC_PI_2, 1.0),
        (core::f32::consts::PI, 0.0),
    ];

    for &(x, expected) in cases {
        let mut backend = EspBackend;
        let q_input = quantize(
            x,
            models::SineModel::input_scale(),
            models::SineModel::input_zero_point(),
        );
        let input = [q_input; models::SineModel::input_len()];
        let mut output = [0i8; models::SineModel::output_len()];

        match models::SineModel::predict_quantized(&mut backend, &input, &mut output) {
            Ok(()) => {
                let result = dequantize(
                    output[0],
                    models::SineModel::output_scale(),
                    models::SineModel::output_zero_point(),
                );
                let err = (result - expected).abs();
                let pass = err < 0.1;
                info!(
                    "[sine] x={} expected={} got={} err={} {}",
                    x,
                    expected,
                    result,
                    err,
                    if pass { "PASS" } else { "FAIL" }
                );
                if !pass {
                    panic!("sine test failed");
                }
            }
            Err(_e) => {
                info!("[sine] inference error");
                panic!("sine inference failed");
            }
        }
    }
}

fn bench_sine() {
    let mut backend = EspBackend;
    let q_input = quantize(
        core::f32::consts::FRAC_PI_2,
        models::SineModel::input_scale(),
        models::SineModel::input_zero_point(),
    );
    let input = [q_input; models::SineModel::input_len()];
    let mut output = [0i8; models::SineModel::output_len()];

    // 预热 1000 次确保 ICache 完全热起来
    for _ in 0..1000 {
        models::SineModel::predict_quantized(&mut backend, &input, &mut output)
            .expect("sine warmup inference failed");
        core::hint::black_box(output[0]);
    }

    let started_at = Instant::now();
    let mut checksum = 0i32;
    for _ in 0..SINE_BENCH_ITERS {
        models::SineModel::predict_quantized(&mut backend, &input, &mut output)
            .expect("sine benchmark inference failed");
        checksum = checksum.wrapping_add(output[0] as i32);
        core::hint::black_box(output[0]);
    }
    let elapsed = started_at.elapsed();
    let elapsed_us = elapsed.as_micros();
    let avg_ns = (elapsed_us * 1_000) / SINE_BENCH_ITERS as u64;

    info!(
        "[sine bench] iters={} elapsed_us={} avg_ns={} checksum={}",
        SINE_BENCH_ITERS, elapsed_us, avg_ns, checksum
    );
}

#[esp_rtos::main]
async fn main(_spawner: Spawner) -> ! {
    // generator version: 1.3.0
    // generator parameters: --chip esp32s3 -o esp32s3-wroom-1-octal-psram -o unstable-hal -o alloc -o wifi -o embassy -o probe-rs -o defmt -o panic-rtt-target -o embedded-test -o ci -o vscode -o esp

    rtt_target::rtt_init_defmt!();

    let config = esp_hal::Config::default().with_cpu_clock(CpuClock::max());
    let peripherals = esp_hal::init(config);

    // The following pins are used to bootstrap the chip. They are available
    // for use, but check the datasheet of the module for more information on them.
    // - GPIO0
    // - GPIO3
    // - GPIO45
    // - GPIO46
    // These GPIO pins are in use by some feature of the module and should not be used.
    let _ = peripherals.GPIO27;
    let _ = peripherals.GPIO28;
    let _ = peripherals.GPIO29;
    let _ = peripherals.GPIO30;
    let _ = peripherals.GPIO31;
    let _ = peripherals.GPIO32;
    let _ = peripherals.GPIO33;
    let _ = peripherals.GPIO34;
    let _ = peripherals.GPIO35;
    let _ = peripherals.GPIO36;
    let _ = peripherals.GPIO37;

    esp_alloc::heap_allocator!(#[esp_hal::ram(reclaimed)] size: 73744);

    let timg0 = TimerGroup::new(peripherals.TIMG0);
    let sw_interrupt =
        esp_hal::interrupt::software::SoftwareInterruptControl::new(peripherals.SW_INTERRUPT);
    esp_rtos::start(timg0.timer0, sw_interrupt.software_interrupt0);

    info!("Embassy initialized!");

    info!("=== ember-esp-nn inference tests ===");

    test_sine();
    bench_sine();

    info!("=== SINE TEST PASSED ===");

    loop {
        Timer::after(Duration::from_secs(5)).await;
    }

    // for inspiration have a look at the examples at https://github.com/esp-rs/esp-hal/tree/esp-hal-v1.1.0/examples
}

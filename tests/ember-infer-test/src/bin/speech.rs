#![no_std]
#![no_main]
#![deny(
    clippy::mem_forget,
    reason = "mem::forget is generally not safe to do with esp_hal types, especially those \
    holding buffers for the duration of a data transfer."
)]
#![deny(clippy::large_stack_frames)]

extern crate alloc;
extern crate self as microflow;
extern crate self as nalgebra;

use alloc::vec;
use defmt::info;
use embassy_executor::Spawner;
use embassy_time::{Duration, Instant, Timer};
use esp_hal::clock::CpuClock;
use esp_hal::timer::timg::TimerGroup;
use panic_rtt_target as _;

esp_bootloader_esp_idf::esp_app_desc!();

use ember_esp_nn::EspBackend;

const SPEECH_BENCH_ITERS: usize = 100;
const SILENCE: usize = 0;
const UNKNOWN: usize = 1;
const YES: usize = 2;
const NO: usize = 3;

#[allow(
    clippy::large_stack_frames,
    reason = "the model macro generates static model metadata for the hardware test"
)]
mod models {
    use ember_infer_macros::model;

    #[model("models/speech.tflite")]
    pub struct SpeechModel;
}

#[macro_export]
macro_rules! matrix {
    ($($value:expr),* $(,)?) => {
        [[$($value),*]]
    };
}

pub mod buffer {
    pub type Buffer2D<T, const ROWS: usize, const COLS: usize> = [[T; COLS]; ROWS];
}

mod samples {
    #![allow(unused_imports)]

    include!("../../../../samples/features/speech.rs");
}

static YES_SAMPLE: [i8; models::SpeechModel::input_len()] = samples::YES[0];
static NO_SAMPLE: [i8; models::SpeechModel::input_len()] = samples::NO[0];

fn class_name(class: usize) -> &'static str {
    match class {
        SILENCE => "silence",
        UNKNOWN => "unknown",
        YES => "yes",
        NO => "no",
        _ => "invalid",
    }
}

fn score(val: i8) -> f32 {
    (val as i32 - models::SpeechModel::output_zero_point()) as f32
        * models::SpeechModel::output_scale()
}

fn argmax(output: &[i8]) -> usize {
    let mut best_index = 0;
    let mut best_value = i8::MIN;

    for (index, value) in output.iter().copied().enumerate() {
        if value > best_value {
            best_index = index;
            best_value = value;
        }
    }

    best_index
}

fn test_speech() {
    let cases: &[(&str, &[i8], usize)] = &[
        ("samples/yes.wav", &YES_SAMPLE, YES),
        ("samples/no.wav", &NO_SAMPLE, NO),
    ];

    for &(name, input, expected) in cases {
        let mut backend = EspBackend;
        let mut scratch = vec![0u8; models::SpeechModel::scratch_len::<EspBackend>()];
        let mut output = [0i8; models::SpeechModel::output_len()];

        match models::SpeechModel::predict_quantized_with_scratch(
            &mut backend,
            input,
            &mut output,
            &mut scratch,
        ) {
            Ok(()) => {
                let predicted = argmax(&output);
                let pass = predicted == expected;
                info!(
                    "[speech] sample={} expected={} got={} raw=[{}, {}, {}, {}] score=[{}, {}, {}, {}] {}",
                    name,
                    class_name(expected),
                    class_name(predicted),
                    output[0],
                    output[1],
                    output[2],
                    output[3],
                    score(output[0]),
                    score(output[1]),
                    score(output[2]),
                    score(output[3]),
                    if pass { "PASS" } else { "FAIL" }
                );
                if !pass {
                    panic!("speech test failed");
                }
            }
            Err(_e) => {
                info!("[speech] inference error");
                panic!("speech inference failed");
            }
        }
    }
}

fn bench_speech() {
    let mut backend = EspBackend;
    let input = &YES_SAMPLE;
    let mut scratch = vec![0u8; models::SpeechModel::scratch_len::<EspBackend>()];
    let mut output = [0i8; models::SpeechModel::output_len()];

    for _ in 0..10 {
        models::SpeechModel::predict_quantized_with_scratch(
            &mut backend,
            input,
            &mut output,
            &mut scratch,
        )
        .expect("speech warmup inference failed");
        core::hint::black_box(output);
    }

    let started_at = Instant::now();
    let mut checksum = 0i32;
    for _ in 0..SPEECH_BENCH_ITERS {
        models::SpeechModel::predict_quantized_with_scratch(
            &mut backend,
            input,
            &mut output,
            &mut scratch,
        )
        .expect("speech benchmark inference failed");
        for value in output {
            checksum = checksum.wrapping_add(value as i32);
        }
        core::hint::black_box(output);
    }
    let elapsed = started_at.elapsed();
    let elapsed_us = elapsed.as_micros();
    let avg_ns = (elapsed_us * 1_000) / SPEECH_BENCH_ITERS as u64;

    info!(
        "[speech bench] iters={} elapsed_us={} avg_ns={} checksum={}",
        SPEECH_BENCH_ITERS, elapsed_us, avg_ns, checksum
    );
}

#[esp_rtos::main]
async fn main(_spawner: Spawner) -> ! {
    rtt_target::rtt_init_defmt!();

    let config = esp_hal::Config::default().with_cpu_clock(CpuClock::max());
    let peripherals = esp_hal::init(config);

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
    info!("=== ember-esp-nn speech inference test ===");

    test_speech();
    bench_speech();

    info!("=== SPEECH TEST PASSED ===");

    loop {
        Timer::after(Duration::from_secs(5)).await;
    }
}

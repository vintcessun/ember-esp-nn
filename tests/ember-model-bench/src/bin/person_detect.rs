#![no_std]
#![no_main]
#![deny(
    clippy::mem_forget,
    reason = "mem::forget is generally not safe to do with esp_hal types, especially those \
    holding buffers for the duration of a data transfer."
)]
#![deny(clippy::large_stack_frames)]

extern crate self as microflow;
extern crate self as nalgebra;

use core::mem::MaybeUninit;

use defmt::info;
use ember_esp_nn::EspBackend;
use esp_hal::{
    clock::CpuClock,
    main,
    time::{Duration, Instant},
};
use panic_rtt_target as _;

esp_bootloader_esp_idf::esp_app_desc!();

const PERSON_BENCH_ITERS: usize = 20;
const NO_PERSON: usize = 0;
const PERSON: usize = 1;
const PERSON_DETECT_SCRATCH_CAP: usize = 72 * 1024;

#[repr(C, align(16))]
struct AlignedScratch([MaybeUninit<u8>; PERSON_DETECT_SCRATCH_CAP]);

unsafe impl esp_hal::Uninit for AlignedScratch {}

#[allow(
    clippy::large_stack_frames,
    reason = "the model macro generates static model metadata for the hardware test"
)]
mod models {
    use ember_infer_macros::model;

    #[model("models/person_detect.tflite")]
    pub struct PersonDetectModel;
}

#[macro_export]
macro_rules! matrix {
    ($($([$value:expr]),+ $(,)?);+;) => {
        [$([$([$value]),+]),+]
    };
    ($($([$value:expr]),+ $(,)?);+) => {
        [$([$([$value]),+]),+]
    };
}

pub mod buffer {
    pub type Buffer4D<T, const N: usize, const H: usize, const W: usize, const C: usize> =
        [[[[T; C]; W]; H]; N];
}

mod samples {
    #![allow(unused_imports)]

    include!("../../../../samples/features/person_detect.rs");
}

static PERSON_SAMPLE: [[[[i8; 1]; 96]; 96]; 1] = samples::PERSON;
static NO_PERSON_SAMPLE: [[[[i8; 1]; 96]; 96]; 1] = samples::NO_PERSON;
#[esp_hal::ram(reclaimed)]
static mut PERSON_DETECT_SCRATCH: AlignedScratch =
    AlignedScratch([MaybeUninit::uninit(); PERSON_DETECT_SCRATCH_CAP]);

fn sample_as_input(sample: &'static [[[[i8; 1]; 96]; 96]; 1]) -> &'static [i8] {
    let ptr = sample.as_ptr().cast::<i8>();
    let len = models::PersonDetectModel::input_len();

    // Buffer4D is a contiguous NHWC array, which matches the generated model input layout.
    unsafe { core::slice::from_raw_parts(ptr, len) }
}

fn class_name(class: usize) -> &'static str {
    match class {
        NO_PERSON => "no_person",
        PERSON => "person",
        _ => "invalid",
    }
}

fn score(val: i8) -> f32 {
    (val as i32 - models::PersonDetectModel::output_zero_point()) as f32
        * models::PersonDetectModel::output_scale()
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

fn run_person_detect_case(name: &str, input: &[i8], expected: usize, scratch: &mut [u8]) {
    let mut backend = EspBackend;
    let mut output = [0i8; models::PersonDetectModel::output_len()];

    models::PersonDetectModel::predict_quantized_with_scratch(
        &mut backend,
        input,
        &mut output,
        scratch,
    )
    .expect("person_detect inference failed");

    let predicted = argmax(&output);
    let pass = predicted == expected;
    info!(
        "[person_detect] sample={} expected={} got={} raw=[{}, {}] score=[{}, {}] {}",
        name,
        class_name(expected),
        class_name(predicted),
        output[0],
        output[1],
        score(output[0]),
        score(output[1]),
        if pass { "PASS" } else { "FAIL" }
    );

    if !pass {
        panic!("person_detect test failed");
    }
}

fn test_person_detect(scratch: &mut [u8]) {
    run_person_detect_case(
        "samples/person.bmp",
        sample_as_input(&PERSON_SAMPLE),
        PERSON,
        scratch,
    );
    run_person_detect_case(
        "samples/no_person.bmp",
        sample_as_input(&NO_PERSON_SAMPLE),
        NO_PERSON,
        scratch,
    );
}

fn bench_person_detect(scratch: &mut [u8]) {
    let mut backend = EspBackend;
    let mut output = [0i8; models::PersonDetectModel::output_len()];

    for _ in 0..3 {
        models::PersonDetectModel::predict_quantized_with_scratch(
            &mut backend,
            sample_as_input(&PERSON_SAMPLE),
            &mut output,
            scratch,
        )
        .expect("person_detect warmup inference failed");
        core::hint::black_box(output);
    }

    let started_at = Instant::now();
    let mut checksum = 0i32;
    for _ in 0..PERSON_BENCH_ITERS {
        models::PersonDetectModel::predict_quantized_with_scratch(
            &mut backend,
            sample_as_input(&PERSON_SAMPLE),
            &mut output,
            scratch,
        )
        .expect("person_detect benchmark inference failed");
        for value in output {
            checksum = checksum.wrapping_add(value as i32);
        }
        core::hint::black_box(output);
    }

    let elapsed_us = started_at.elapsed().as_micros();
    let avg_us = elapsed_us / PERSON_BENCH_ITERS as u64;
    let avg_ms = avg_us / 1_000;
    let avg_ms_frac = avg_us % 1_000;

    info!(
        "[person_detect bench] iters={} elapsed_ms={} avg_ms={}.{:03} checksum={}",
        PERSON_BENCH_ITERS,
        elapsed_us / 1_000,
        avg_ms,
        avg_ms_frac,
        checksum
    );
}

fn person_detect_scratch() -> &'static mut [u8] {
    let scratch_len = models::PersonDetectModel::scratch_len::<EspBackend>();
    assert!(
        scratch_len <= PERSON_DETECT_SCRATCH_CAP,
        "person_detect scratch too large"
    );

    unsafe {
        let ptr = core::ptr::addr_of_mut!(PERSON_DETECT_SCRATCH.0).cast::<u8>();
        core::slice::from_raw_parts_mut(ptr, scratch_len)
    }
}

#[allow(
    clippy::large_stack_frames,
    reason = "hardware app entry point owns peripheral setup"
)]
#[main]
fn main() -> ! {
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

    let scratch_len = models::PersonDetectModel::scratch_len::<EspBackend>();
    let scratch = person_detect_scratch();

    info!(
        "=== ember-esp-nn person_detect inference test === scratch_len={} scratch_ptr=0x{:x}",
        scratch_len,
        scratch.as_ptr() as usize
    );

    test_person_detect(scratch);
    bench_person_detect(scratch);

    info!("=== PERSON_DETECT TEST PASSED ===");

    loop {
        let delay_start = Instant::now();
        while delay_start.elapsed() < Duration::from_secs(5) {}
    }
}

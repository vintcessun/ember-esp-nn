# ember-esp-nn

ESP-NN-backed inference backend for Ember on Espressif targets.

This repository provides two `no_std` crates:

```text
crates/
  ember_esp_nn/  Ember KernelBackend implementation backed by ESP-NN
  esp_nn_sys/     Raw Rust FFI bindings and native ESP-NN build layer
```

`esp_nn_sys` vendors [Espressif ESP-NN](https://github.com/espressif/esp-nn) as a git submodule and builds the selected ESP-NN C/assembly kernels into a static native library. `ember_esp_nn` exposes those kernels through `ember_infer_core::KernelBackend`.

## Supported Targets

| Feature | Target | Native compiler | ESP-NN sources |
| --- | --- | --- | --- |
| `ansi` | Generic supported target | target-appropriate GCC | Portable ANSI kernels |
| `esp32s3` | `xtensa-esp32s3-none-elf` | `xtensa-esp32s3-elf-gcc` | ANSI + ESP32-S3 optimized kernels |
| `esp32p4` | `riscv32imafc-unknown-none-elf` | `riscv32-esp-elf-gcc` | ANSI + ESP32-P4 optimized kernels |

`ember_esp_nn` enables `esp32s3` by default.

## Requirements

- Rust with the target you want to build for.
- ESP Rust toolchain for Xtensa targets.
- ESP GCC toolchains available on `PATH`:
  - `xtensa-esp-elf-gcc` for ANSI builds targeting Xtensa.
  - `xtensa-esp32s3-elf-gcc` for optimized ESP32-S3 builds.
  - `riscv32-esp-elf-gcc` for ESP32-P4 builds.
- `libclang` available for `bindgen`.

## Clone

Clone with submodules:

```powershell
git clone --recurse-submodules https://github.com/vintcessun/ember-esp-nn.git
```

For an existing clone:

```powershell
git submodule update --init --recursive
```

## Build

ESP32-S3:

```powershell
cargo check -p ember_esp_nn --target xtensa-esp32s3-none-elf --features esp32s3
```

ESP32-P4:

```powershell
cargo check -p ember_esp_nn --target riscv32imafc-unknown-none-elf --features esp32p4
```

ANSI:

```powershell
cargo check -p ember_esp_nn --features ansi
```

## Hardware Test

ESP32-S3 hardware inference test:

```powershell
cargo build -p ember-infer-test
cd tests\ember-infer-test
cargo run
```

The test binary uses `EspBackend` with:

```text
models/sine.tflite
```

Output is emitted through defmt/RTT. The firmware prints sine `PASS` lines and reports the elapsed runtime when the inference test finishes.

## Benchmarks

Benchmarks are measured on ESP32-S3 at 240 MHz. Outputs are checked against ANSI reference kernels. Basic math kernels use one untimed warm-up call before profiling to reduce first-call cache effects.

| Kernel | ANSI cycles | Optimized cycles | Speedup |
| --- | ---: | ---: | ---: |
| `add_s8` | 304,782 | 60,945 | 5.00x |
| `mul_s8` | 129,403 | 29,451 | 4.39x |
| `mul_broadcast_ch_s8` | 243,195 | 55,811 | 4.35x |
| `depthwise_conv_s8` | 58,366 | 11,443 | 5.10x |
| `conv_s8` | 8,539,821 | 708,197 | 12.05x |
| `relu6_s8` | 1,121 | 92 | 12.18x |
| `avg_pool_s8` | 425,958 | 118,199 | 3.60x |
| `max_pool_s8` | 396,945 | 48,109 | 8.25x |
| `fc_s8` | 837 | 757 | 1.10x |
| `fc_per_ch_s8` | 1,294 | 1,159 | 1.11x |
| `softmax_s8` | 14,630 | 10,814 | 1.35x |
| `hard_swish_s8` | 865,617 | 94,826 | 9.12x |
| `mean_nhwc_s8` | 13,243 | 12,525 | 1.05x |

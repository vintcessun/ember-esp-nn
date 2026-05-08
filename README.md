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

- **Chip:** ESP32-S3
- **CPU:** 240 MHz
- **Runner:** probe-rs + RTT/defmt
- **Memory:** internal RAM

Outputs are checked against ANSI reference kernels. Basic math kernels use one untimed warm-up call before profiling.

`SKIP_NUDGE` is enabled by default for esp32s3 builds (matching IDF production `CONFIG_NN_SKIP_NUDGE=y`), using the faster non-bit-exact re-quantize path. Suppress with `CONFIG_NN_SKIP_NUDGE=false`.

`conv_s8`, `depthwise_conv_s8`, and `fc` kernels are reported per case to match the official ESP-NN benchmark format; re-run the harness to populate the `—` entries.

`conv_s8/8×8,64×3×3×3` uses a 10×10 input with pad=0 stride=1, producing an 8×8 output — "8×8" is the **output** dimension, consistent with the official ESP-NN labeling convention.

| Kernel                | Data info         | ANSI cycles | Optimized cycles | Speedup |
| --------------------- | ----------------- | ----------: | ---------------: | ------: |
| `add_s8`              | 1,615 elements    |     304,225 |           60,651 |   5.01x |
| `mul_s8`              | 1,615 elements    |     130,736 |           29,070 |   4.49x |
| `mul_broadcast_ch_s8` | 49×64             |     242,811 |           55,432 |   4.38x |
| `conv_s8`             | 10×10, 64×1×1×64  |   4,808,830 |          308,753 |  15.57x |
| `conv_s8`             | 8×8, 16×1×1×16    |     328,701 |           32,646 |  10.06x |
| `conv_s8`             | 8×8, 64×3×3×3     |   2,505,725 |          393,156 |   6.37x |
| `depthwise_conv_s8`   | 18×18, 1×3×3×16   |   1,181,492 |          156,762 |   7.53x |
| `depthwise_conv_s8`   | 12×12, 8×5×5×4    |   1,734,451 |          379,681 |   4.56x |
| `relu6_s8`            | 100 elements      |       1,123 |               92 |  12.20x |
| `avg_pool_s8`         | 16×16×16, 3×3     |     425,906 |          118,072 |   3.60x |
| `max_pool_s8`         | 16×16×16, 3×3     |     396,945 |           48,110 |   8.25x |
| `fc_s8`               | 271 row, 3 out ch |       1,081 |            1,005 |   1.07x |
| `fc_per_ch_s8`        | 271 row, 3 out ch |       1,295 |            1,159 |   1.11x |
| `softmax_s8`          | 8×3               |      12,742 |            9,732 |   1.30x |
| `hard_swish_s8`       | 12,544 elements   |     865,617 |           94,826 |   9.12x |
| `mean_nhwc_s8`        | 3×3×96            |      13,239 |           12,521 |   1.05x |

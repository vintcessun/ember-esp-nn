# burn_esp_nn

Rust bindings and integration work for [Espressif ESP-NN](https://github.com/espressif/esp-nn).

This workspace currently contains a working `no_std` sys crate, `esp_nn_sys`, which generates Rust FFI bindings for ESP-NN and compiles the matching ESP-NN C/assembly sources for selected ESP targets. The higher-level `burn_esp_nn` crate is reserved for the safe Burn integration layer.

## Workspace Layout

```text
crates/
  esp_nn_sys/     Raw FFI bindings and native ESP-NN build script
  burn_esp_nn/    Future safe integration layer for Burn
```

`esp_nn_sys` vendors ESP-NN as a git submodule at:

```text
crates/esp_nn_sys/vendor/esp-nn
```

## Status

- `esp_nn_sys` builds ESP-NN into a static native library with `cc`.
- Rust bindings are generated from ESP-NN headers with `bindgen`.
- The crate is `#![no_std]`.
- ANSI, ESP32-S3, and ESP32-P4 source selections are supported.
- The final native link is carried through Cargo metadata from the sys crate to downstream binaries.
- `burn_esp_nn` does not expose a high-level API yet.

## Supported Backends

| Feature | Intended target | Native compiler | ESP-NN sources |
| --- | --- | --- | --- |
| `ansi` | Generic supported target | target-appropriate GCC | Portable ANSI ESP-NN sources |
| `esp32s3` | `xtensa-esp32s3-none-elf` | `xtensa-esp32s3-elf-gcc` | ANSI + ESP32-S3 optimized sources |
| `esp32p4` | `riscv32imafc-unknown-none-elf` | `riscv32-esp-elf-gcc` | ANSI + ESP32-P4 optimized sources |

If no feature is selected, `esp_nn_sys` builds the ANSI source set.

## Requirements

- Rust with the target you want to build for.
- The ESP Rust toolchain for Xtensa targets.
- ESP GCC toolchains available on `PATH`:
  - `xtensa-esp-elf-gcc` for ANSI builds targeting Xtensa.
  - `xtensa-esp32s3-elf-gcc` for optimized ESP32-S3 builds.
  - `riscv32-esp-elf-gcc` for ESP32-P4 builds.
- `libclang` available for `bindgen`.

## Clone

Clone with submodules:

```powershell
git clone --recurse-submodules https://github.com/vintcessun/burn_esp_nn.git
```

For an existing clone:

```powershell
git submodule update --init --recursive
```

## Build Checks

ANSI on RISC-V:

```powershell
cargo clippy -p esp_nn_sys --target riscv32imac-unknown-none-elf --features ansi
```

ANSI on Xtensa:

```powershell
cargo clippy -p esp_nn_sys --target xtensa-esp32s3-none-elf --features ansi
```

ESP32-S3 optimized sources:

```powershell
cargo clippy -p esp_nn_sys --target xtensa-esp32s3-none-elf --features esp32s3
```

ESP32-P4 optimized sources:

```powershell
cargo clippy -p esp_nn_sys --target riscv32imafc-unknown-none-elf --features esp32p4
```

## Build Logging

The build script is quiet by default. To confirm the selected compiler, flags, defines, and ESP-NN source files, set:

```powershell
$env:ESP_NN_SYS_VERBOSE='1'
cargo clippy -p esp_nn_sys --target riscv32imafc-unknown-none-elf --features esp32p4
```

## Regenerating `build.rs`

`crates/esp_nn_sys/build.rs` is generated from `crates/esp_nn_sys/build-template.rs` and the vendored ESP-NN `CMakeLists.txt`.

After changing the template or source extraction logic, run:

```powershell
python scripts\generate_esp_nn_sys_build_rs.py
```

Check that the generated file is up to date:

```powershell
python scripts\generate_esp_nn_sys_build_rs.py --check
```

## Packaging `esp_nn_sys`

From the sys crate directory:

```powershell
cd crates\esp_nn_sys
cargo package --allow-dirty
cargo publish --dry-run --allow-dirty
```

The packaged crate includes the required ESP-NN headers and source files under `vendor/esp-nn`, so crates.io users do not need the workspace-level git submodule.

# esp_nn_sys

Raw `no_std` Rust FFI bindings for [Espressif ESP-NN](https://github.com/espressif/esp-nn).

This crate generates bindings to the ESP-NN C API with `bindgen`, compiles the selected ESP-NN C/assembly implementation with `cc`, and exposes the generated symbols directly from Rust.

## What This Crate Provides

- Generated Rust bindings for ESP-NN functions, types, and constants.
- A static native `esp_nn` library built from vendored ESP-NN sources.
- Cargo link metadata for downstream crates and final firmware binaries.
- `#![no_std]` support.
- Source selection for ANSI, ESP32-S3, and ESP32-P4 builds.

This is a sys crate. It exposes raw FFI and does not provide a safe high-level Rust API.

## Features

| Feature | Source set | Intended target |
| --- | --- | --- |
| default | ANSI | Any supported target/toolchain |
| `ansi` | ANSI | Any supported target/toolchain |
| `esp32s3` | ANSI + ESP32-S3 optimized sources | `xtensa-esp32s3-none-elf` |
| `esp32p4` | ANSI + ESP32-P4 optimized sources | `riscv32imafc-unknown-none-elf` |

Only one optimized chip feature should be enabled at a time.

## Toolchain Requirements

You need the Rust target and matching C compiler for the target you are building:

| Build | Default C compiler |
| --- | --- |
| Xtensa ANSI | `xtensa-esp-elf-gcc` |
| ESP32-S3 optimized | `xtensa-esp32s3-elf-gcc` |
| RISC-V ANSI | `riscv32-esp-elf-gcc` |
| ESP32-P4 optimized | `riscv32-esp-elf-gcc` |

`bindgen` also requires `libclang`.

The compiler can be overridden with the usual `cc` crate environment variables, for example:

```powershell
$env:CC_riscv32imafc_unknown_none_elf='riscv32-esp-elf-gcc'
```

## Examples

ANSI build for RISC-V:

```powershell
cargo build --target riscv32imac-unknown-none-elf --features ansi
```

ESP32-S3 optimized build:

```powershell
cargo build --target xtensa-esp32s3-none-elf --features esp32s3
```

ESP32-P4 optimized build:

```powershell
cargo build --target riscv32imafc-unknown-none-elf --features esp32p4
```

## Build Script Behavior

The build script:

1. Locates the vendored ESP-NN source directory.
2. Generates Rust bindings from `wrapper.h`, which includes `esp_nn.h`.
3. Selects the ANSI source list plus the requested optimized chip source list.
4. Defines the matching ESP-IDF target macro for optimized builds.
5. Builds a static `esp_nn` native library.
6. Emits Cargo link metadata for downstream crates.

For ESP32-P4, the build script passes:

```text
-march=rv32imafc_xespv
```

This enables the ESP vector instruction extension used by the optimized P4 assembly.

## Verbose Build Output

The build script is quiet by default. To print the selected compiler, defines, flags, and source files:

```powershell
$env:ESP_NN_SYS_VERBOSE='1'
cargo build --target riscv32imafc-unknown-none-elf --features esp32p4
```

The output is intentionally emitted through Cargo warnings because build script stdout is hidden by Cargo during normal builds.

## Optional Defines

Set `CONFIG_NN_SKIP_NUDGE` to enable ESP-NN's `SKIP_NUDGE` define:

```powershell
$env:CONFIG_NN_SKIP_NUDGE='1'
```

## Safety

All exported functions are raw FFI bindings. Calling them is unsafe when the underlying C API requires valid pointers, aligned buffers, correct tensor dimensions, and target-compatible memory. Higher-level crates should wrap these APIs before exposing them to application code.

## Vendored ESP-NN

The crates.io package includes the required ESP-NN headers and source files under:

```text
vendor/esp-nn
```

Consumers of the published crate do not need to initialize a git submodule.


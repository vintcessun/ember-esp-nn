# Models

Support/verification status of the `.tflite` files in this directory on ESP32-S3
(via the `#[model("...")]` macro + the `ember_esp_nn` backend).

| Model | Test binary | Status | Notes |
|-------|-------------|--------|-------|
| `sine.tflite` | `ember-infer-test-sine` | ✅ Verified on HW | tiny FC regression |
| `speech.tflite` | `ember-infer-test-speech` | ✅ Verified on HW (debug + release) | micro-speech: depthwise → FC → softmax |
| `person_detect.tflite` | `ember-model-bench-person-detect` | ✅ Verified on HW | 14-layer MobileNet-style, 96×96×1 input |
| `mobilenet.tflite` | — | ❌ **Not supported** | see below |

## `mobilenet.tflite` — not runnable (kept for reference only)

This is **MobileNetV3-Small** (224×224×3 input, 1000 classes, 111 operators). It
cannot run with the current library for two independent reasons:

1. **Unsupported operators.** It uses `HARD_SWISH` and `MEAN` (global average
   pool), which the `#[model]` macro does not emit and the `KernelBackend` trait
   does not expose, so the macro aborts at **compile time** with
   `unsupported operator: HARD_SWISH`. (`MUL` *is* now supported — esp-nn
   accelerated, matching esp-tflite-micro. ESP-NN also has `hard_swish` and
   `mean` s8 kernels, so wiring these two up is mechanical — but see #2.)
2. **Memory.** The input is 150 KB and the largest activation tensor is ~1 MB,
   which exceeds the ESP32-S3's 512 KB SRAM. Activation tensors are stack arrays,
   so this model would not fit even if all operators were supported.

### Accelerated operator coverage vs esp-tflite-micro

esp-tflite-micro accelerates exactly these int8 ops with esp-nn: `CONV_2D`,
`DEPTHWISE_CONV_2D`, `FULLY_CONNECTED`, `AVG_POOL_2D`, `MAX_POOL_2D`, `ADD`,
`MUL`, `SOFTMAX`. **This backend now covers all of them** (`MUL` added last). The
other esp-nn s8 kernels (`hard_swish`, `mean`, `logistic`) are *not* wired here —
esp-tflite-micro also leaves those TFLite ops on the reference (non-esp-nn)
kernels, so this is parity, not a gap.

Do not add a test binary for `mobilenet.tflite` until both are addressed (operator
coverage + a streaming/tiled or PSRAM-backed activation strategy).

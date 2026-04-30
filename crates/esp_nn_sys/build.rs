use std::{
    env,
    path::{Path, PathBuf},
};

const C_SRCS: &[&str] = &[
    // C_SRCS
    "src/activation_functions/esp_nn_relu_ansi.c",
    "src/activation_functions/esp_nn_hard_swish_ansi.c",
    "src/common/esp_nn_mean_ansi.c",
    "src/basic_math/esp_nn_add_ansi.c",
    "src/basic_math/esp_nn_mul_ansi.c",
    "src/convolution/esp_nn_conv_ansi.c",
    "src/convolution/esp_nn_conv_opt.c",
    "src/convolution/esp_nn_depthwise_conv_ansi.c",
    "src/convolution/esp_nn_depthwise_conv_opt.c",
    "src/fully_connected/esp_nn_fully_connected_ansi.c",
    "src/softmax/esp_nn_softmax_ansi.c",
    "src/softmax/esp_nn_softmax_opt.c",
    "src/logistic/esp_nn_logistic_ansi.c",
    "src/pooling/esp_nn_avg_pool_ansi.c",
    "src/pooling/esp_nn_max_pool_ansi.c",
];

const ESP32S3_SRCS: &[&str] = &[
    // S3_SRCS
    "src/common/esp_nn_common_functions_esp32s3.S",
    "src/common/esp_nn_dot_s8_esp32s3.S",
    "src/common/esp_nn_multiply_by_quantized_mult_esp32s3.S",
    "src/common/esp_nn_multiply_by_quantized_mult_ver1_esp32s3.S",
    "src/activation_functions/esp_nn_relu_s8_esp32s3.S",
    "src/activation_functions/esp_nn_hard_swish_s8_esp32s3.c",
    "src/common/esp_nn_mean_s8_esp32s3.c",
    "src/basic_math/esp_nn_add_s8_esp32s3.S",
    "src/basic_math/esp_nn_mul_s8_esp32s3.S",
    "src/basic_math/esp_nn_mul_broadcast_s8_esp32s3.S",
    "src/convolution/esp_nn_conv_esp32s3.c",
    "src/convolution/esp_nn_conv_s8_1x1_esp32s3.c",
    "src/convolution/esp_nn_conv_s8_3x3_opt_esp32s3.c",
    "src/convolution/esp_nn_depthwise_conv_s8_esp32s3.c",
    "src/convolution/esp_nn_conv_s16_mult8_esp32s3.S",
    "src/convolution/esp_nn_conv_s8_mult8_1x1_esp32s3.S",
    "src/convolution/esp_nn_conv_s16_mult4_1x1_esp32s3.S",
    "src/convolution/esp_nn_conv_s8_filter_aligned_input_padded_esp32s3.S",
    "src/convolution/esp_nn_depthwise_conv_s8_mult1_3x3_padded_esp32s3.S",
    "src/convolution/esp_nn_depthwise_conv_s16_mult1_esp32s3.S",
    "src/convolution/esp_nn_depthwise_conv_s16_mult1_3x3_esp32s3.S",
    "src/convolution/esp_nn_depthwise_conv_s16_mult1_3x3_no_pad_esp32s3.S",
    "src/convolution/esp_nn_depthwise_conv_s16_mult8_3x3_esp32s3.S",
    "src/convolution/esp_nn_depthwise_conv_s16_mult4_esp32s3.S",
    "src/convolution/esp_nn_depthwise_conv_s16_mult8_esp32s3.S",
    "src/fully_connected/esp_nn_fully_connected_esp32s3.c",
    "src/fully_connected/esp_nn_fc_s8_mac16_esp32s3.S",
    "src/fully_connected/esp_nn_fully_connected_s8_esp32s3.S",
    "src/fully_connected/esp_nn_fully_connected_per_ch_s8_esp32s3.S",
    "src/pooling/esp_nn_max_pool_s8_esp32s3.S",
    "src/pooling/esp_nn_avg_pool_s8_esp32s3.c",
    "src/pooling/esp_nn_avg_pool_s8_esp32s3.S",
    "src/softmax/esp_nn_softmax_s8_esp32s3.c",
];

const ESP32P4_SRCS: &[&str] = &[
    // P4_SRCS
    "src/common/esp_nn_mean_s8_esp32p4.c",
    "src/common/esp_nn_multiply_by_quantized_mult_esp32p4.S",
    "src/activation_functions/esp_nn_hard_swish_s8_esp32p4.c",
    "src/activation_functions/esp_nn_relu_s8_esp32p4.c",
    "src/basic_math/esp_nn_add_s8_esp32p4.c",
    "src/basic_math/esp_nn_mul_s8_esp32p4.c",
    "src/convolution/esp_nn_conv_esp32p4.c",
    "src/convolution/esp_nn_depthwise_conv_esp32p4.c",
    "src/fully_connected/esp_nn_fully_connected_s8_esp32p4.c",
    "src/pooling/esp_nn_avg_pool_s8_esp32p4.c",
    "src/pooling/esp_nn_max_pool_s8_esp32p4.c",
    "src/softmax/esp_nn_softmax_s8_esp32p4.c",
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Chip {
    Ansi,
    Esp32s3,
    Esp32p4,
}

fn chip_name(chip: Chip) -> &'static str {
    match chip {
        Chip::Ansi => "ansi",
        Chip::Esp32s3 => "esp32s3",
        Chip::Esp32p4 => "esp32p4",
    }
}

fn detect_chip() -> Chip {
    let target = env::var("TARGET").unwrap_or_default();

    if cfg!(feature = "esp32s3") {
        if !target.contains(chip_name(Chip::Esp32s3)) {
            panic!(
                "feature 'esp32s3' is not compatible with target '{target}' (target should contains '{}')",
                chip_name(Chip::Esp32s3)
            );
        }
        Chip::Esp32s3
    } else if cfg!(feature = "esp32p4") {
        Chip::Esp32p4
    } else {
        Chip::Ansi
    }
}

fn cc_env_var_names(target: &str) -> [String; 4] {
    let target_env = target.replace('-', "_");

    [
        format!("CC_{target}"),
        format!("CC_{target_env}"),
        "TARGET_CC".to_owned(),
        "CC".to_owned(),
    ]
}

fn has_user_configured_cc(target: &str) -> bool {
    cc_env_var_names(target)
        .into_iter()
        .any(|key| env::var_os(key).is_some())
}

fn env_flag_enabled(key: &str) -> bool {
    env::var(key)
        .map(|value| {
            let value = value.to_ascii_lowercase();
            !matches!(value.as_str(), "" | "0" | "false" | "n" | "no" | "off")
        })
        .unwrap_or(false)
}

fn default_compiler(target: &str, chip: Chip) -> Option<&'static str> {
    if target.starts_with("xtensa-") {
        if chip == Chip::Esp32s3 {
            return Some("xtensa-esp32s3-elf-gcc");
        }
        return Some("xtensa-esp-elf-gcc");
    }
    if target.starts_with("riscv32") {
        return Some("riscv32-esp-elf-gcc");
    }
    None
}

fn log_build_step(message: impl AsRef<str>) {
    if env_flag_enabled("ESP_NN_SYS_VERBOSE") {
        println!("cargo:warning=esp-nn build: {}", message.as_ref());
    }
}

fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let esp_nn_dir = find_esp_nn_dir(&manifest_dir);
    let include_dir = esp_nn_dir.join("include");
    let common_dir = esp_nn_dir.join("src").join("common");

    println!("cargo:rerun-if-changed=wrapper.h");
    println!("cargo:rerun-if-changed={}", esp_nn_dir.display());

    let chip = detect_chip();

    generate_bindings(&include_dir, &common_dir, chip);
    compile_esp_nn(&esp_nn_dir, &include_dir, &common_dir, chip);
}

fn find_esp_nn_dir(manifest_dir: &Path) -> PathBuf {
    let packaged_dir = manifest_dir.join("vendor").join("esp-nn");
    if packaged_dir.exists() {
        return packaged_dir;
    }

    manifest_dir
        .ancestors()
        .nth(2)
        .expect("esp_nn_sys should be under crates/esp_nn_sys")
        .join("vendor")
        .join("esp-nn")
}

fn generate_bindings(include_dir: &Path, common_dir: &Path, chip: Chip) {
    let host = env::var("HOST").unwrap();

    let mut builder = bindgen::Builder::default()
        .header("wrapper.h")
        .clang_arg(format!("--target={host}"))
        .clang_arg(format!("-I{}", include_dir.display()))
        .clang_arg(format!("-I{}", common_dir.display()))
        .use_core()
        .ctypes_prefix("core::ffi")
        .layout_tests(false)
        .allowlist_function("esp_nn_.*")
        .allowlist_type(".*_t")
        .allowlist_var("ESP_NN_.*");

    match chip {
        Chip::Esp32s3 => {
            builder = builder.clang_arg("-DCONFIG_IDF_TARGET_ESP32S3=1");
        }
        Chip::Esp32p4 => {
            builder = builder.clang_arg("-DCONFIG_IDF_TARGET_ESP32P4=1");
        }
        Chip::Ansi => {}
    }

    let bindings = builder.generate().expect("failed to generate bindings");

    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("failed to write bindings");
}

fn compile_esp_nn(esp_nn_dir: &Path, include_dir: &Path, common_dir: &Path, chip: Chip) {
    let mut build = cc::Build::new();
    let target = env::var("TARGET").unwrap_or_default();
    let mut compile_flags: Vec<&str> = vec!["-O2", "-Wno-unused-function"];
    let mut compile_defines: Vec<&str> = Vec::new();

    for key in cc_env_var_names(&target) {
        println!("cargo:rerun-if-env-changed={key}");
    }
    println!("cargo:rerun-if-env-changed=CFLAGS");
    println!("cargo:rerun-if-env-changed=TARGET_CFLAGS");
    println!("cargo:rerun-if-env-changed=CFLAGS_{target}");
    println!("cargo:rerun-if-env-changed=CONFIG_NN_SKIP_NUDGE");
    println!("cargo:rerun-if-env-changed=ESP_NN_SYS_VERBOSE");
    println!(
        "cargo:rerun-if-env-changed=CFLAGS_{}",
        target.replace('-', "_")
    );

    log_build_step(format!("target={target}, chip={}", chip_name(chip)));
    log_build_step(format!("include={}", include_dir.display()));
    log_build_step(format!("include={}", common_dir.display()));

    if !has_user_configured_cc(&target)
        && let Some(compiler) = default_compiler(&target, chip)
    {
        log_build_step(format!("compiler={compiler} (default)"));
        build.compiler(compiler);
    } else {
        log_build_step("compiler selected by cc/Cargo environment");
    }

    build
        .include(include_dir)
        .include(common_dir)
        .warnings(false)
        .flag_if_supported("-O2")
        .flag_if_supported("-Wno-unused-function");

    if env_flag_enabled("CONFIG_NN_SKIP_NUDGE") {
        compile_defines.push("SKIP_NUDGE");
        build.define("SKIP_NUDGE", None);
    }

    match chip {
        Chip::Esp32s3 => {
            compile_defines.push("CONFIG_IDF_TARGET_ESP32S3=1");
            compile_flags.push("-mlongcalls");
            compile_flags.push("-fno-unroll-loops");
            build
                .define("CONFIG_IDF_TARGET_ESP32S3", Some("1"))
                .flag_if_supported("-mlongcalls")
                .flag_if_supported("-fno-unroll-loops");
        }
        Chip::Esp32p4 => {
            compile_defines.push("CONFIG_IDF_TARGET_ESP32P4=1");
            compile_flags.push("-march=rv32imafc_xespv");
            build
                .define("CONFIG_IDF_TARGET_ESP32P4", Some("1"))
                .flag_if_supported("-march=rv32imafc_xespv");
        }
        Chip::Ansi => {}
    }

    if compile_defines.is_empty() {
        log_build_step("defines=(none)");
    } else {
        log_build_step(format!("defines={}", compile_defines.join(" ")));
    }
    log_build_step(format!("flags={}", compile_flags.join(" ")));

    let mut files: Vec<(&str, &str)> = Vec::new();
    files.extend(C_SRCS.iter().copied().map(|file| ("C_SRCS", file)));

    match chip {
        Chip::Esp32s3 => files.extend(
            ESP32S3_SRCS
                .iter()
                .copied()
                .map(|file| ("ESP32S3_SRCS", file)),
        ),
        Chip::Esp32p4 => files.extend(
            ESP32P4_SRCS
                .iter()
                .copied()
                .map(|file| ("ESP32P4_SRCS", file)),
        ),
        Chip::Ansi => {}
    }

    log_build_step(format!("compiling {} source files", files.len()));

    for (group, file) in files {
        let path = esp_nn_dir.join(file);
        if !path.exists() {
            panic!("ESP-NN source not found: {}", path.display());
        }

        println!("cargo:rerun-if-changed={}", path.display());
        log_build_step(format!("file[{group}]={}", path.display()));
        build.file(path);
    }

    build.compile("esp_nn");
}

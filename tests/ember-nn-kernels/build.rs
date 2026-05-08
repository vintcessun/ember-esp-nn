fn main() {
    generate_esp_nn_test_vectors();
    linker_be_nice();
    println!("cargo:rustc-link-arg=-Tdefmt.x");
    // make sure linkall.x is the last linker script (otherwise might cause problems with flip-link)
    println!("cargo:rustc-link-arg=-Tlinkall.x");
}

fn generate_esp_nn_test_vectors() {
    let manifest_dir =
        std::path::PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").expect("manifest dir"));
    let repo_root = manifest_dir
        .ancestors()
        .nth(2)
        .expect("tests/ember-nn-kernels should be under repo/tests");
    let conv_test = repo_root
        .join("crates")
        .join("esp_nn_sys")
        .join("vendor")
        .join("esp-nn")
        .join("tests")
        .join("src")
        .join("convolution_test.c");
    let basic_math_test = repo_root
        .join("crates")
        .join("esp_nn_sys")
        .join("vendor")
        .join("esp-nn")
        .join("tests")
        .join("src")
        .join("basic_math_test.c");
    println!("cargo:rerun-if-changed={}", conv_test.display());
    println!("cargo:rerun-if-changed={}", basic_math_test.display());

    let src = std::fs::read_to_string(&conv_test).expect("read convolution_test.c");
    let basic_math_src = std::fs::read_to_string(&basic_math_test).expect("read basic_math_test.c");
    let mut out = String::new();
    out.push_str("// Generated from esp-nn/tests/src/*.c.\n");
    out.push_str("// Keep these vectors sourced from vendor C to avoid transcription drift.\n\n");
    write_array(
        &mut out,
        &basic_math_src,
        "int8_t",
        "test_add_in1",
        "i8",
        "TEST_ADD_IN1",
    );
    write_array(
        &mut out,
        &basic_math_src,
        "int8_t",
        "test_add_in2",
        "i8",
        "TEST_ADD_IN2",
    );
    write_array(&mut out, &src, "int8_t", "yolo_filter", "i8", "YOLO_FILTER");
    write_array(&mut out, &src, "int8_t", "yolo_input", "i8", "YOLO_INPUT");
    write_array(&mut out, &src, "int32_t", "yolo_bias", "i32", "YOLO_BIAS");
    write_array(
        &mut out,
        &src,
        "int32_t",
        "yolo_shifts",
        "i32",
        "YOLO_SHIFTS",
    );
    write_array(&mut out, &src, "int32_t", "yolo_mults", "i32", "YOLO_MULTS");

    let out_dir = std::path::PathBuf::from(std::env::var("OUT_DIR").expect("OUT_DIR"));
    std::fs::write(out_dir.join("esp_nn_test_vectors.rs"), out)
        .expect("write esp_nn_test_vectors.rs");
}

fn write_array(
    out: &mut String,
    src: &str,
    c_type: &str,
    c_name: &str,
    rust_type: &str,
    rust_name: &str,
) {
    let needle = format!("static const {c_type} {c_name}[] = {{");
    let fallback_needle = format!("const {c_type} {c_name}[] = {{");
    let (start, needle_len) = src
        .find(&needle)
        .map(|start| (start, needle.len()))
        .or_else(|| {
            src.find(&fallback_needle)
                .map(|start| (start, fallback_needle.len()))
        })
        .expect("array start");
    let start = start + needle_len;
    let rest = &src[start..];
    let end = rest.find("};").expect("array end");
    let body = &rest[..end];
    let values: Vec<&str> = body
        .split(',')
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .collect();

    out.push_str(&format!(
        "const {rust_name}: [{rust_type}; {}] = [\n",
        values.len()
    ));
    for chunk in values.chunks(16) {
        out.push_str("    ");
        out.push_str(&chunk.join(", "));
        out.push_str(",\n");
    }
    out.push_str("];\n\n");
}

fn linker_be_nice() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() > 1 {
        let kind = &args[1];
        let what = &args[2];

        match kind.as_str() {
            "undefined-symbol" => match what.as_str() {
                what if what.starts_with("_defmt_") => {
                    eprintln!();
                    eprintln!(
                        "💡 `defmt` not found - make sure `defmt.x` is added as a linker script and you have included `use defmt_rtt as _;`"
                    );
                    eprintln!();
                }
                "_stack_start" => {
                    eprintln!();
                    eprintln!("💡 Is the linker script `linkall.x` missing?");
                    eprintln!();
                }
                what if what.starts_with("esp_rtos_") => {
                    eprintln!();
                    eprintln!(
                        "💡 `esp-radio` has no scheduler enabled. Make sure you have initialized `esp-rtos` or provided an external scheduler."
                    );
                    eprintln!();
                }
                "embedded_test_linker_file_not_added_to_rustflags" => {
                    eprintln!();
                    eprintln!(
                        "💡 `embedded-test` not found - make sure `embedded-test.x` is added as a linker script for tests"
                    );
                    eprintln!();
                }
                "free"
                | "malloc"
                | "calloc"
                | "get_free_internal_heap_size"
                | "malloc_internal"
                | "realloc_internal"
                | "calloc_internal"
                | "free_internal" => {
                    eprintln!();
                    eprintln!(
                        "💡 Did you forget the `esp-alloc` dependency or didn't enable the `compat` feature on it?"
                    );
                    eprintln!();
                }
                _ => (),
            },
            // we don't have anything helpful for "missing-lib" yet
            _ => {
                std::process::exit(1);
            }
        }

        std::process::exit(0);
    }

    println!(
        "cargo:rustc-link-arg=-Wl,--error-handling-script={}",
        std::env::current_exe().unwrap().display()
    );
}

fn main() {
    cc::Build::new()
        .file("src/stdio_stubs.c")
        .warnings(false)
        .compile("ember_esp_nn_stdio_stubs");
}

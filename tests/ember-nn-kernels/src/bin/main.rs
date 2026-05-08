#![no_std]
#![no_main]
#![allow(clippy::large_stack_frames)]
#![feature(asm_experimental_arch)]

use defmt::info;
use esp_hal::clock::CpuClock;
use esp_hal::delay::Delay;
use esp_hal::main;
use panic_rtt_target as _;

// Always import both ansi AND esp32s3 variants explicitly by full name
use esp_nn_sys::bindings::{
    act_params_t,
    conv_params_t,
    data_2d_t,
    // structs
    data_dims_t,
    dw_conv_params_t,
    esp_nn_add_elementwise_s8_ansi,
    esp_nn_add_elementwise_s8_esp32s3,
    esp_nn_avg_pool_s8_ansi,
    esp_nn_avg_pool_s8_esp32s3,
    esp_nn_conv_s8_ansi,
    esp_nn_conv_s8_esp32s3,
    esp_nn_depthwise_conv_s8_ansi,
    esp_nn_depthwise_conv_s8_esp32s3,
    esp_nn_fully_connected_per_ch_s8_ansi,
    esp_nn_fully_connected_per_ch_s8_esp32s3,
    // ansi — reference implementations
    esp_nn_fully_connected_s8_ansi,
    // esp32s3 — optimized implementations
    esp_nn_fully_connected_s8_esp32s3,
    esp_nn_get_conv_scratch_size_esp32s3,
    esp_nn_get_depthwise_conv_scratch_size_esp32s3,
    esp_nn_get_hard_swish_scratch_size_esp32s3,
    esp_nn_get_softmax_scratch_size_esp32s3,
    esp_nn_hard_swish_s8_ansi,
    esp_nn_hard_swish_s8_esp32s3,
    esp_nn_max_pool_s8_ansi,
    esp_nn_max_pool_s8_esp32s3,
    esp_nn_mean_nhwc_s8_ansi,
    esp_nn_mean_nhwc_s8_esp32s3,
    esp_nn_mul_broadcast_channel_s8_ansi,
    esp_nn_mul_broadcast_channel_s8_esp32s3,
    esp_nn_mul_elementwise_s8_ansi,
    esp_nn_mul_elementwise_s8_esp32s3,
    esp_nn_relu6_s8_ansi,
    esp_nn_relu6_s8_esp32s3,
    esp_nn_set_conv_scratch_buf_esp32s3,
    esp_nn_set_depthwise_conv_scratch_buf_esp32s3,
    esp_nn_set_hard_swish_scratch_buf_esp32s3,
    esp_nn_set_softmax_scratch_buf_esp32s3,
    esp_nn_softmax_s8_ansi,
    esp_nn_softmax_s8_esp32s3,
    quant_data_t,
};

esp_bootloader_esp_idf::esp_app_desc!();

include!(concat!(env!("OUT_DIR"), "/esp_nn_test_vectors.rs"));

// --- Cycle counter ---
#[inline(always)]
fn ccount() -> u32 {
    let v: u32;
    unsafe { core::arch::asm!("rsr.ccount {0}", out(reg) v) };
    v
}

#[unsafe(no_mangle)]
extern "C" fn puts(_s: *const core::ffi::c_char) -> core::ffi::c_int {
    0
}

const MULT_MAX: i32 = i32::MAX;
const MULT_MIN: i32 = 0;
const SHIFT_MIN: i32 = -31;
const SHIFT_MAX: i32 = 30;

unsafe extern "C" {
    fn malloc(size: usize) -> *mut core::ffi::c_void;
    fn free(ptr: *mut core::ffi::c_void);
}

fn fc_mult(row_len: u16, rng: &mut NewlibRand) -> i32 {
    (MULT_MAX / row_len as i32).wrapping_add(rng.rand() % i16::MAX as i32)
}

#[repr(C, align(16))]
struct Aligned<T, const N: usize>([T; N]);

impl<T: Copy, const N: usize> Aligned<T, N> {
    fn new(value: T) -> Self {
        Self([value; N])
    }

    fn slice_mut(&mut self, len: usize) -> &mut [T] {
        &mut self.0[..len]
    }
}

impl<T, const N: usize> core::ops::Deref for Aligned<T, N> {
    type Target = [T; N];

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T, const N: usize> core::ops::DerefMut for Aligned<T, N> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

struct TestMalloc {
    raw: *mut i8,
}

impl TestMalloc {
    fn new(size: usize) -> Self {
        let raw = unsafe { malloc(size) as *mut i8 };
        if raw.is_null() {
            panic!("ESP_NN_TEST_ALLOC failed");
        }
        Self { raw }
    }

    fn aligned_slice_mut(&mut self, len: usize) -> &mut [i8] {
        let aligned = ((self.raw as usize + 15) & !15) as *mut i8;
        unsafe { core::slice::from_raw_parts_mut(aligned, len) }
    }
}

impl Drop for TestMalloc {
    fn drop(&mut self) {
        unsafe { free(self.raw as *mut core::ffi::c_void) };
    }
}

/// Exact replica of newlib's rand() used by ESP-IDF / xtensa toolchain.
/// Formula: next = next * 1103515245 + 12345
/// Returns: (next >> 16) & 0x7fff  (range 0..=32767)
/// Default seed (no srand call): 1
struct NewlibRand {
    state: u32,
}

impl NewlibRand {
    /// Create with the same default seed as C's uninitialized rand() — seed=1
    fn new() -> Self {
        Self { state: 1 }
    }

    /// Equivalent to C's rand()
    fn rand(&mut self) -> i32 {
        self.state = self.state.wrapping_mul(1103515245).wrapping_add(12345);
        ((self.state >> 16) & 0x7fff) as i32
    }

    /// rand() % 256 - 128  (same as C tests for i8 data)
    fn rand_i8(&mut self) -> i8 {
        (self.rand() % 256 - 128) as i8
    }

    /// rand() % 256 - 128 variant used in some tests
    fn rand_i8_255(&mut self) -> i8 {
        (self.rand() % 255 - 128) as i8
    }

    /// rand() % 128 (used in depthwise conv input)
    fn rand_i8_128(&mut self) -> i8 {
        (self.rand() % 128) as i8
    }

    /// rand() % 32767 (used for bias/mult data)
    fn rand_i32_int16(&mut self) -> i32 {
        self.rand() % 32767
    }

    /// rand() % 65535 + 255 (used for conv bias)
    fn rand_u16_plus_u8(&mut self) -> i32 {
        self.rand() % 65535 + 255
    }

    /// -10 + rand() % 2 (used for conv shift)
    fn rand_shift_conv(&mut self) -> i32 {
        -10 + self.rand() % 2
    }

    /// 0x7f67f4f8 + rand() % 50 (used for conv mult)
    fn rand_mult_conv(&mut self) -> i32 {
        0x7f67f4f8i32.wrapping_add(self.rand() % 50)
    }

    fn range(&mut self, modulus: i32, offset: i32) -> i32 {
        self.rand() % modulus + offset
    }
}

fn report_profile(name: &str, c_ansi: u32, c_opt: u32) {
    let sx100 = if c_opt > 0 {
        (c_ansi as u64 * 100 / c_opt as u64) as u32
    } else {
        0
    };
    info!(
        "PROFILE: {}, ansi={}, opt={}, speedup={}.{}{}x",
        name,
        c_ansi,
        c_opt,
        sx100 / 100,
        (sx100 / 10) % 10,
        sx100 % 10
    );
    log_delay();
}

struct ProfileSummary {
    last_c_ansi: u32,
    last_c_opt: u32,
}

impl ProfileSummary {
    fn new() -> Self {
        Self {
            last_c_ansi: 0,
            last_c_opt: 0,
        }
    }

    fn record(
        &mut self,
        name: &str,
        iter: usize,
        c_ansi: u32,
        c_opt: u32,
        out_ansi: &[i8],
        out_opt: &[i8],
    ) {
        verify_outputs(name, iter, out_ansi, out_opt);
        self.last_c_ansi = c_ansi;
        self.last_c_opt = c_opt;
    }

    fn print(&self, name: &str) {
        report_profile(name, self.last_c_ansi, self.last_c_opt);
    }
}

fn verify_outputs(name: &str, iter: usize, out_ansi: &[i8], out_opt: &[i8]) -> bool {
    let passed = out_ansi == out_opt;
    if !passed {
        // find first differing index
        let first_diff = out_ansi
            .iter()
            .zip(out_opt.iter())
            .position(|(a, b)| a != b);
        if let Some(idx) = first_diff {
            let start = idx.saturating_sub(2);
            let end = (idx + 6).min(out_ansi.len());
            info!("  first diff at index={} (out of {})", idx, out_ansi.len());
            log_delay();
            info!("  ansi[{}..{}]: {:?}", start, end, &out_ansi[start..end]);
            log_delay();
            info!("  opt [{}..{}]: {:?}", start, end, &out_opt[start..end]);
            log_delay();
        }
        panic!("{} iter={} outputs differ", name, iter);
    }
    passed
}

fn log_section(text: &str) {
    info!("{}", text);
    log_delay();
}

fn log_delay() {
    let delay = Delay::new();
    delay.delay_millis(50);
}

fn test_fully_connected(rng: &mut NewlibRand) {
    const MAX_ROW_LEN: usize = 271;
    const MAX_OUT_CH: usize = 16;

    let input_offset: i32 = 0;
    let filter_offset: i32 = 0;
    let out_offset: i32 = 5;
    let act_min: i32 = -128;
    let act_max: i32 = 127;

    let mut row_len: u16 = 271;
    let mut out_ch: u16 = 3;
    let mut input_buf = Aligned::<i8, MAX_ROW_LEN>([0; MAX_ROW_LEN]);
    let mut filter_buf = Aligned::<i8, { MAX_ROW_LEN * MAX_OUT_CH }>([0; MAX_ROW_LEN * MAX_OUT_CH]);
    let mut out_ansi_buf = Aligned::<i8, MAX_OUT_CH>([0; MAX_OUT_CH]);
    let mut out_opt_buf = Aligned::<i8, MAX_OUT_CH>([0; MAX_OUT_CH]);
    let mut profile = ProfileSummary::new();

    for iter in 0..15 {
        let out_mult = fc_mult(row_len, rng);
        let out_shift = match iter {
            0 => SHIFT_MAX,
            1 => SHIFT_MIN,
            2 => SHIFT_MAX,
            3 => 0,
            4 => {
                row_len = 1;
                out_ch = 16;
                -10 + rng.rand() % 5
            }
            5 => {
                row_len = 16;
                out_ch = 8;
                -10 + rng.rand() % 5
            }
            6 => {
                row_len = 8;
                out_ch = 8;
                -10 + rng.rand() % 5
            }
            7 => {
                row_len = 8;
                out_ch = 15;
                -10 + rng.rand() % 5
            }
            8 => {
                row_len = 8;
                out_ch = 1;
                -10 + rng.rand() % 5
            }
            _ => {
                row_len = (rng.rand() % 7 + 1) as u16;
                out_ch = 8;
                -10 + rng.rand() % 5
            }
        };

        let input_len = row_len as usize;
        let filter_len = row_len as usize * out_ch as usize;
        let out_len = out_ch as usize;
        let input = &mut input_buf[..input_len];
        let filter = &mut filter_buf[..filter_len];
        let out_ansi = &mut out_ansi_buf[..out_len];
        let out_opt = &mut out_opt_buf[..out_len];
        out_ansi.fill(0);
        out_opt.fill(0);
        for v in input.iter_mut() {
            *v = rng.rand_i8();
        }
        for v in filter.iter_mut() {
            *v = rng.rand_i8();
        }

        let t0 = ccount();
        unsafe {
            esp_nn_fully_connected_s8_ansi(
                input.as_ptr(),
                input_offset,
                row_len,
                filter.as_ptr(),
                filter_offset,
                core::ptr::null(),
                out_ansi.as_mut_ptr(),
                out_ch,
                out_offset,
                out_shift,
                out_mult,
                act_min,
                act_max,
            );
        }
        let c_ansi = ccount().wrapping_sub(t0);

        let t0 = ccount();
        unsafe {
            esp_nn_fully_connected_s8_esp32s3(
                input.as_ptr(),
                input_offset,
                row_len,
                filter.as_ptr(),
                filter_offset,
                core::ptr::null(),
                out_opt.as_mut_ptr(),
                out_ch,
                out_offset,
                out_shift,
                out_mult,
                act_min,
                act_max,
            );
        }
        let c_opt = ccount().wrapping_sub(t0);

        profile.record("fc_s8", iter, c_ansi, c_opt, out_ansi, out_opt);
    }

    profile.print("fc_s8");
}

fn test_fully_connected_per_ch(rng: &mut NewlibRand) {
    const MAX_ROW_LEN: usize = 271;
    const MAX_OUT_CH: usize = 16;

    let input_offset: i32 = 0;
    let filter_offset: i32 = 0;
    let out_offset: i32 = 7;
    let act_min: i32 = -128;
    let act_max: i32 = 127;

    let mut row_len: u16 = 271;
    let mut out_ch: u16 = 3;
    let mut input_buf = Aligned::<i8, MAX_ROW_LEN>([0; MAX_ROW_LEN]);
    let mut filter_buf = Aligned::<i8, { MAX_ROW_LEN * MAX_OUT_CH }>([0; MAX_ROW_LEN * MAX_OUT_CH]);
    let mut out_ansi_buf = Aligned::<i8, MAX_OUT_CH>([0; MAX_OUT_CH]);
    let mut out_opt_buf = Aligned::<i8, MAX_OUT_CH>([0; MAX_OUT_CH]);
    let mut out_mult_buf = Aligned::<i32, MAX_OUT_CH>([0; MAX_OUT_CH]);
    let mut out_shift_buf = Aligned::<i32, MAX_OUT_CH>([0; MAX_OUT_CH]);
    let mut profile = ProfileSummary::new();

    for iter in 0..15 {
        let out_shift_val = match iter {
            0 => -10,
            1 => SHIFT_MIN,
            2 => SHIFT_MAX,
            3 => 0,
            4 => {
                row_len = 1;
                out_ch = 16;
                0
            }
            5 => {
                row_len = 16;
                out_ch = 8;
                0
            }
            6 => {
                row_len = 8;
                out_ch = 8;
                0
            }
            7 => {
                row_len = 8;
                out_ch = 15;
                0
            }
            8 => {
                row_len = 8;
                out_ch = 1;
                0
            }
            _ => {
                row_len = (rng.rand() % 7 + 1) as u16;
                out_ch = 8;
                0
            }
        };

        let out_mult = &mut out_mult_buf[..out_ch as usize];
        let out_shift = &mut out_shift_buf[..out_ch as usize];
        for i in 0..out_ch as usize {
            out_mult[i] = fc_mult(row_len, rng);
            out_shift[i] = if i < 4 {
                out_shift_val
            } else {
                -10 + rng.rand() % 5
            };
        }

        let input_len = row_len as usize;
        let filter_len = row_len as usize * out_ch as usize;
        let out_len = out_ch as usize;
        let input = &mut input_buf[..input_len];
        let filter = &mut filter_buf[..filter_len];
        let out_ansi = &mut out_ansi_buf[..out_len];
        let out_opt = &mut out_opt_buf[..out_len];
        out_ansi.fill(0);
        out_opt.fill(0);
        for v in input.iter_mut() {
            *v = rng.rand_i8();
        }
        for v in filter.iter_mut() {
            *v = rng.rand_i8();
        }

        let t0 = ccount();
        unsafe {
            esp_nn_fully_connected_per_ch_s8_ansi(
                input.as_ptr(),
                input_offset,
                row_len,
                filter.as_ptr(),
                filter_offset,
                core::ptr::null(),
                out_ansi.as_mut_ptr(),
                out_ch,
                out_offset,
                out_shift.as_mut_ptr(),
                out_mult.as_mut_ptr(),
                act_min,
                act_max,
            );
        }
        let c_ansi = ccount().wrapping_sub(t0);

        let t0 = ccount();
        unsafe {
            esp_nn_fully_connected_per_ch_s8_esp32s3(
                input.as_ptr(),
                input_offset,
                row_len,
                filter.as_ptr(),
                filter_offset,
                core::ptr::null(),
                out_opt.as_mut_ptr(),
                out_ch,
                out_offset,
                out_shift.as_mut_ptr(),
                out_mult.as_mut_ptr(),
                act_min,
                act_max,
            );
        }
        let c_opt = ccount().wrapping_sub(t0);

        profile.record("fc_per_ch_s8", iter, c_ansi, c_opt, out_ansi, out_opt);
    }

    profile.print("fc_per_ch_s8");
}

fn test_add(rng: &mut NewlibRand) {
    let act_min: i32 = -128;
    let act_max: i32 = 127;

    let mut size = 1615;
    let mut profile = ProfileSummary::new();

    for iter in 0..10 {
        let (
            i1_offset,
            i2_offset,
            out_offset,
            i1_mult,
            i2_mult,
            out_mult,
            i1_shift,
            i2_shift,
            out_shift,
            left_shift,
        ) = match iter {
            0 => (0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
            1 => (-127, -127, -128, MULT_MIN, MULT_MIN, MULT_MIN, 0, 0, 0, 0),
            2 => (
                128, 128, -127, MULT_MAX, MULT_MAX, MULT_MAX, SHIFT_MIN, SHIFT_MIN, SHIFT_MIN, 22,
            ),
            3 => (128, 128, -127, MULT_MAX, MULT_MAX, MULT_MAX, 0, 0, 0, 22),
            4 => {
                size = 216;
                (
                    64, 128, -128, 1705397815, 1073741824, 1756091225, -3, 0, -19, 20,
                )
            }
            _ => (
                rng.range(256, -127),
                rng.range(256, -127),
                rng.range(256, -128),
                MULT_MAX / 2 + rng.rand() % i16::MAX as i32,
                MULT_MAX / 2 + rng.rand() % i16::MAX as i32,
                MULT_MAX / 2 + rng.rand() % i16::MAX as i32,
                -8 + rng.rand() % 4,
                -8 + rng.rand() % 4,
                -8 + rng.rand() % 4,
                rng.rand() % 15,
            ),
        };

        let len = size as usize;
        let mut in1_buf = TestMalloc::new(len + 16);
        let mut in2_buf = TestMalloc::new(len + 16);
        let mut out_ansi_buf = TestMalloc::new(len + 16);
        let mut out_opt_buf = TestMalloc::new(len + 16);

        let in1 = in1_buf.aligned_slice_mut(len);
        let in2 = in2_buf.aligned_slice_mut(len);
        let out_ansi = out_ansi_buf.aligned_slice_mut(len);
        let out_opt = out_opt_buf.aligned_slice_mut(len);
        if iter == 4 {
            in1.copy_from_slice(&TEST_ADD_IN1);
            in2.copy_from_slice(&TEST_ADD_IN2);
        } else {
            for i in 0..len {
                in1[i] = rng.rand_i8();
                in2[i] = rng.rand_i8();
            }
        }

        let t0 = if iter == 0 { ccount() } else { 0 };
        unsafe {
            esp_nn_add_elementwise_s8_ansi(
                in1.as_ptr(),
                in2.as_ptr(),
                i1_offset,
                i2_offset,
                i1_mult,
                i2_mult,
                i1_shift,
                i2_shift,
                left_shift,
                out_ansi.as_mut_ptr(),
                out_offset,
                out_mult,
                out_shift,
                act_min,
                act_max,
                size,
            );
        }
        let c_ansi = if iter == 0 {
            ccount().wrapping_sub(t0)
        } else {
            0
        };

        if iter == 0 {
            unsafe {
                esp_nn_add_elementwise_s8_esp32s3(
                    in1.as_ptr(),
                    in2.as_ptr(),
                    i1_offset,
                    i2_offset,
                    i1_mult,
                    i2_mult,
                    i1_shift,
                    i2_shift,
                    left_shift,
                    out_opt.as_mut_ptr(),
                    out_offset,
                    out_mult,
                    out_shift,
                    act_min,
                    act_max,
                    size,
                );
            }
        }

        let t0 = if iter == 0 { ccount() } else { 0 };
        unsafe {
            esp_nn_add_elementwise_s8_esp32s3(
                in1.as_ptr(),
                in2.as_ptr(),
                i1_offset,
                i2_offset,
                i1_mult,
                i2_mult,
                i1_shift,
                i2_shift,
                left_shift,
                out_opt.as_mut_ptr(),
                out_offset,
                out_mult,
                out_shift,
                act_min,
                act_max,
                size,
            );
        }
        let c_opt = if iter == 0 {
            ccount().wrapping_sub(t0)
        } else {
            0
        };

        if iter == 0 {
            profile.record("add_s8", iter, c_ansi, c_opt, out_ansi, out_opt);
        } else {
            verify_outputs("add_s8", iter, out_ansi, out_opt);
        }
    }

    profile.print("add_s8");
}

fn test_mul(rng: &mut NewlibRand) {
    let act_min: i32 = -128;
    let act_max: i32 = 127;
    let mut profile = ProfileSummary::new();

    for iter in 0..10 {
        let mut size = 1615;
        let (i1_offset, i2_offset, out_offset, out_mult, out_shift) = match iter {
            0 => (0, 0, 0, 0, 0),
            1 => (-127, -127, -128, MULT_MIN, 0),
            2 => (128, 128, -127, MULT_MAX, SHIFT_MIN),
            3 => (128, 128, -127, MULT_MAX, 0),
            _ => {
                let i1_offset = rng.range(256, -127);
                let i2_offset = rng.range(256, -127);
                let out_offset = rng.range(256, -128);
                let out_mult = MULT_MAX / 2 + rng.rand() % i16::MAX as i32;
                let out_shift = -8 + rng.rand() % 4;
                size = 4 + rng.rand() % 64;
                (i1_offset, i2_offset, out_offset, out_mult, out_shift)
            }
        };

        let len = size as usize;
        let mut in1_buf = TestMalloc::new(len + 16);
        let mut in2_buf = TestMalloc::new(len + 16);
        let mut out_ansi_buf = TestMalloc::new(len + 16);
        let mut out_opt_buf = TestMalloc::new(len + 16);

        let in1 = in1_buf.aligned_slice_mut(len);
        let in2 = in2_buf.aligned_slice_mut(len);
        let out_ansi = out_ansi_buf.aligned_slice_mut(len);
        let out_opt = out_opt_buf.aligned_slice_mut(len);
        for i in 0..len {
            in1[i] = rng.rand_i8();
            in2[i] = rng.rand_i8();
        }

        let t0 = if iter == 0 { ccount() } else { 0 };
        unsafe {
            esp_nn_mul_elementwise_s8_ansi(
                in1.as_ptr(),
                in2.as_ptr(),
                i1_offset,
                i2_offset,
                out_ansi.as_mut_ptr(),
                out_offset,
                out_mult,
                out_shift,
                act_min,
                act_max,
                size,
            );
        }
        let c_ansi = if iter == 0 {
            ccount().wrapping_sub(t0)
        } else {
            0
        };

        if iter == 0 {
            unsafe {
                esp_nn_mul_elementwise_s8_esp32s3(
                    in1.as_ptr(),
                    in2.as_ptr(),
                    i1_offset,
                    i2_offset,
                    out_opt.as_mut_ptr(),
                    out_offset,
                    out_mult,
                    out_shift,
                    act_min,
                    act_max,
                    size,
                );
            }
        }

        let t0 = if iter == 0 { ccount() } else { 0 };
        unsafe {
            esp_nn_mul_elementwise_s8_esp32s3(
                in1.as_ptr(),
                in2.as_ptr(),
                i1_offset,
                i2_offset,
                out_opt.as_mut_ptr(),
                out_offset,
                out_mult,
                out_shift,
                act_min,
                act_max,
                size,
            );
        }
        let c_opt = if iter == 0 {
            ccount().wrapping_sub(t0)
        } else {
            0
        };

        if iter == 0 {
            profile.record("mul_s8", iter, c_ansi, c_opt, out_ansi, out_opt);
        } else {
            verify_outputs("mul_s8", iter, out_ansi, out_opt);
        }
    }

    profile.print("mul_s8");
}

fn test_mul_broadcast_channel(rng: &mut NewlibRand) {
    let act_min: i32 = -128;
    let act_max: i32 = 127;
    let mut total_spatial = 49;
    let mut channels = 64;
    let mut i1_offset = 34;
    let mut i2_offset = 35;
    let mut out_offset = 36;
    let mut out_shift = -7;
    let mut out_mult = MULT_MAX;
    let mut profile = ProfileSummary::new();

    for iter in 0..10 {
        match iter {
            0 => {
                i1_offset = 0;
                i2_offset = 0;
                out_offset = 0;
                out_mult = 0;
                out_shift = 0;
                total_spatial = 49;
                channels = 64;
            }
            1 => {
                i1_offset = -127;
                i2_offset = -127;
                out_offset = -128;
                out_mult = MULT_MIN;
                out_shift = 0;
            }
            2 => {
                i1_offset = 128;
                i2_offset = 128;
                out_offset = -127;
                out_mult = MULT_MAX;
                out_shift = SHIFT_MIN;
            }
            3 => {
                i1_offset = 64;
                i2_offset = 32;
                out_offset = -10;
                out_mult = MULT_MAX / 2;
                out_shift = -5;
                total_spatial = 16;
                channels = 5;
            }
            4 => {
                total_spatial = 14;
                channels = 19;
            }
            5 => {
                i1_offset = 128;
                i2_offset = 128;
                out_offset = -128;
                out_mult = 1705397815;
                out_shift = -3;
                total_spatial = 49;
                channels = 96;
            }
            _ => {
                i1_offset = rng.range(256, -127);
                i2_offset = rng.range(256, -127);
                out_offset = rng.range(256, -128);
                out_mult = MULT_MAX / 2 + rng.rand() % i16::MAX as i32;
                out_shift = -8 + rng.rand() % 4;
                total_spatial = 4 + rng.rand() % 64;
                channels = 8 + rng.rand() % 128;
            }
        };
        let size = total_spatial * channels;
        let len = size as usize;
        let ch_len = channels as usize;
        let mut in1_buf = TestMalloc::new(len + 16);
        let mut in2_buf = TestMalloc::new(ch_len + 16);
        let mut out_ansi_buf = TestMalloc::new(len + 16);
        let mut out_opt_buf = TestMalloc::new(len + 16);

        let in1 = in1_buf.aligned_slice_mut(len);
        let in2 = in2_buf.aligned_slice_mut(ch_len);
        let out_ansi = out_ansi_buf.aligned_slice_mut(len);
        let out_opt = out_opt_buf.aligned_slice_mut(len);
        for v in in1.iter_mut() {
            *v = rng.rand_i8();
        }
        for v in in2.iter_mut() {
            *v = rng.rand_i8();
        }

        let t0 = if iter == 0 { ccount() } else { 0 };
        unsafe {
            esp_nn_mul_broadcast_channel_s8_ansi(
                in1.as_ptr(),
                in2.as_ptr(),
                i1_offset,
                i2_offset,
                out_ansi.as_mut_ptr(),
                out_offset,
                out_mult,
                out_shift,
                act_min,
                act_max,
                total_spatial,
                channels,
            );
        }
        let c_ansi = if iter == 0 {
            ccount().wrapping_sub(t0)
        } else {
            0
        };

        if iter == 0 {
            unsafe {
                esp_nn_mul_broadcast_channel_s8_esp32s3(
                    in1.as_ptr(),
                    in2.as_ptr(),
                    i1_offset,
                    i2_offset,
                    out_opt.as_mut_ptr(),
                    out_offset,
                    out_mult,
                    out_shift,
                    act_min,
                    act_max,
                    total_spatial,
                    channels,
                );
            }
        }

        let t0 = if iter == 0 { ccount() } else { 0 };
        unsafe {
            esp_nn_mul_broadcast_channel_s8_esp32s3(
                in1.as_ptr(),
                in2.as_ptr(),
                i1_offset,
                i2_offset,
                out_opt.as_mut_ptr(),
                out_offset,
                out_mult,
                out_shift,
                act_min,
                act_max,
                total_spatial,
                channels,
            );
        }
        let c_opt = if iter == 0 {
            ccount().wrapping_sub(t0)
        } else {
            0
        };

        if iter == 0 {
            profile.record(
                "mul_broadcast_ch_s8",
                iter,
                c_ansi,
                c_opt,
                out_ansi,
                out_opt,
            );
        } else {
            verify_outputs("mul_broadcast_ch_s8", iter, out_ansi, out_opt);
        }
    }

    profile.print("mul_broadcast_ch_s8");
}

fn test_avg_pool(rng: &mut NewlibRand) {
    let cases = &[
        (16, 16, 16, 3, 3, 1, 1, 1, 1),
        (16, 16, 4, 3, 3, 1, 1, 1, 1),
        (16, 16, 8, 3, 3, 1, 1, 1, 1),
        (16, 16, 32, 3, 3, 1, 1, 1, 1),
        (16, 16, 64, 3, 3, 1, 1, 1, 1),
        (16, 16, 16, 1, 1, 1, 1, 0, 0),
        (16, 16, 16, 2, 2, 1, 1, 0, 0),
        (16, 16, 16, 5, 5, 1, 1, 2, 2),
        (16, 16, 16, 3, 3, 2, 2, 1, 1),
        (24, 24, 32, 3, 3, 2, 2, 1, 1),
        (6, 6, 128, 6, 6, 1, 1, 0, 0),
        (16, 16, 16, 3, 3, 1, 1, 0, 0),
    ];
    let act_min: i32 = -128;
    let act_max: i32 = 127;
    let mut profile = ProfileSummary::new();

    for (iter, &(iwd, iht, ch, fwd, fht, swd, sht, pwd, pht)) in cases.iter().enumerate() {
        let out_wd = (iwd + 2 * pwd - fwd) / swd + 1;
        let out_ht = (iht + 2 * pht - fht) / sht + 1;
        let in_size = iwd as usize * iht as usize * ch as usize;
        let out_size = out_wd as usize * out_ht as usize * ch as usize;

        let mut input_buf = TestMalloc::new(in_size + 16);
        let mut out_ansi_buf = TestMalloc::new(out_size + 16);
        let mut out_opt_buf = TestMalloc::new(out_size + 16);

        let input = input_buf.aligned_slice_mut(in_size);
        let out_ansi = out_ansi_buf.aligned_slice_mut(out_size);
        let out_opt = out_opt_buf.aligned_slice_mut(out_size);
        out_ansi.fill(0);
        out_opt.fill(0);
        for v in input.iter_mut() {
            *v = rng.rand_i8();
        }

        let t0 = ccount();
        unsafe {
            esp_nn_avg_pool_s8_ansi(
                input.as_ptr(),
                iwd,
                iht,
                out_ansi.as_mut_ptr(),
                out_wd,
                out_ht,
                swd,
                sht,
                fwd,
                fht,
                pwd,
                pht,
                act_min,
                act_max,
                ch,
            );
        }
        let c_ansi = ccount().wrapping_sub(t0);

        let t0 = ccount();
        unsafe {
            esp_nn_avg_pool_s8_esp32s3(
                input.as_ptr(),
                iwd,
                iht,
                out_opt.as_mut_ptr(),
                out_wd,
                out_ht,
                swd,
                sht,
                fwd,
                fht,
                pwd,
                pht,
                act_min,
                act_max,
                ch,
            );
        }
        let c_opt = ccount().wrapping_sub(t0);

        profile.record("avg_pool_s8", iter, c_ansi, c_opt, out_ansi, out_opt);
    }

    profile.print("avg_pool_s8");
}

fn test_max_pool(rng: &mut NewlibRand) {
    let cases = &[
        (16, 16, 16, 3, 3, 1, 1, 1, 1),
        (16, 16, 4, 3, 3, 1, 1, 1, 1),
        (16, 16, 8, 3, 3, 1, 1, 1, 1),
        (16, 16, 32, 3, 3, 1, 1, 1, 1),
        (16, 16, 64, 3, 3, 1, 1, 1, 1),
        (16, 16, 16, 1, 1, 1, 1, 0, 0),
        (16, 16, 16, 2, 2, 1, 1, 0, 0),
        (16, 16, 16, 5, 5, 1, 1, 2, 2),
        (16, 16, 16, 3, 3, 2, 2, 1, 1),
        (24, 24, 32, 3, 3, 2, 2, 1, 1),
        (6, 6, 128, 6, 6, 1, 1, 0, 0),
        (16, 16, 16, 3, 3, 1, 1, 0, 0),
    ];
    let act_min: i32 = -128;
    let act_max: i32 = 127;

    let mut profile = ProfileSummary::new();

    for (iter, &(iwd, iht, ch, fwd, fht, swd, sht, pwd, pht)) in cases.iter().enumerate() {
        let out_wd = (iwd + 2 * pwd - fwd) / swd + 1;
        let out_ht = (iht + 2 * pht - fht) / sht + 1;
        let in_size = iwd as usize * iht as usize * ch as usize;
        let out_size = out_wd as usize * out_ht as usize * ch as usize;

        let mut input_buf = TestMalloc::new(in_size + 16);
        let mut out_ansi_buf = TestMalloc::new(out_size + 16);
        let mut out_opt_buf = TestMalloc::new(out_size + 16);

        let input = input_buf.aligned_slice_mut(in_size);
        let out_ansi = out_ansi_buf.aligned_slice_mut(out_size);
        let out_opt = out_opt_buf.aligned_slice_mut(out_size);
        out_ansi.fill(0);
        out_opt.fill(0);
        for v in input.iter_mut() {
            *v = rng.rand_i8();
        }

        let t0 = ccount();
        unsafe {
            esp_nn_max_pool_s8_ansi(
                input.as_ptr(),
                iwd,
                iht,
                out_ansi.as_mut_ptr(),
                out_wd,
                out_ht,
                swd,
                sht,
                fwd,
                fht,
                pwd,
                pht,
                act_min,
                act_max,
                ch,
            );
        }
        let c_ansi = ccount().wrapping_sub(t0);

        let t0 = ccount();
        unsafe {
            esp_nn_max_pool_s8_esp32s3(
                input.as_ptr(),
                iwd,
                iht,
                out_opt.as_mut_ptr(),
                out_wd,
                out_ht,
                swd,
                sht,
                fwd,
                fht,
                pwd,
                pht,
                act_min,
                act_max,
                ch,
            );
        }
        let c_opt = ccount().wrapping_sub(t0);

        profile.record("max_pool_s8", iter, c_ansi, c_opt, out_ansi, out_opt);
    }

    profile.print("max_pool_s8");
}

fn test_softmax(rng: &mut NewlibRand) {
    const MAX_SIZE: usize = 64 * 32;
    const MAX_SCRATCH_SIZE: usize = 64 * 32 * 4;

    let cases: &[(i32, i32, i32, i32, i32)] = &[
        (8, 32, i32::MAX / 2, 7, -128),
        (1, 2, i32::MAX / 2, 7, -128),
        (1, 4, i32::MAX / 2, 7, -128),
        (1, 1, i32::MAX / 2, 7, -128),
        (1, 10, i32::MAX / 2, 7, -128),
        (4, 10, i32::MAX / 2, 7, -128),
        (1, 1000, i32::MAX / 2, 7, -128),
        (64, 32, i32::MAX / 2, 7, -128),
        (8, 32, i32::MAX / 2, 7, -64),
        (8, 32, i32::MAX / 2, 7, -32),
        (8, 32, i32::MAX / 2, 7, 0),
        (8, 32, i32::MAX / 4, 5, -128),
        (8, 32, i32::MAX, 10, -128),
        (8, 17, i32::MAX / 2, 7, -128),
        (8, 3, i32::MAX / 2, 7, -128),
    ];

    let mut input_buf = Aligned::<i8, MAX_SIZE>::new(0);
    let mut out_ansi_buf = Aligned::<i8, MAX_SIZE>::new(0);
    let mut out_opt_buf = Aligned::<i8, MAX_SIZE>::new(0);
    let mut scratch_buf = Aligned::<u8, MAX_SCRATCH_SIZE>::new(0);
    let mut profile = ProfileSummary::new();

    for (iter, &(height, width, mult, shift, diff_min)) in cases.iter().enumerate() {
        let size = (height * width) as usize;
        let input = input_buf.slice_mut(size);
        let out_ansi = out_ansi_buf.slice_mut(size);
        let out_opt = out_opt_buf.slice_mut(size);
        out_ansi.fill(0);
        out_opt.fill(0);
        for v in input.iter_mut() {
            *v = rng.rand_i8_255();
        }

        let t0 = ccount();
        unsafe {
            esp_nn_softmax_s8_ansi(
                input.as_ptr(),
                height,
                width,
                mult,
                shift,
                diff_min,
                out_ansi.as_mut_ptr(),
            );
        }
        let c_ansi = ccount().wrapping_sub(t0);

        let scratch_size =
            unsafe { esp_nn_get_softmax_scratch_size_esp32s3(width, height) } as usize;
        let scratch = scratch_buf.slice_mut(scratch_size.max(1));
        unsafe {
            esp_nn_set_softmax_scratch_buf_esp32s3(scratch.as_mut_ptr() as *mut core::ffi::c_void);
        }

        let t0 = ccount();
        unsafe {
            esp_nn_softmax_s8_esp32s3(
                input.as_ptr(),
                height,
                width,
                mult,
                shift,
                diff_min,
                out_opt.as_mut_ptr(),
            );
        }
        let c_opt = ccount().wrapping_sub(t0);

        profile.record("softmax_s8", iter, c_ansi, c_opt, out_ansi, out_opt);
    }

    profile.print("softmax_s8");
}

fn test_relu6(rng: &mut NewlibRand) {
    const MAX_SIZE: usize = 1615;

    let sizes: &[u16] = &[1615, 1, 3, 7, 8, 12, 15, 16, 32, 256, 17, 33, 100];
    let mut out_ansi_buf = Aligned::<i8, MAX_SIZE>::new(0);
    let mut out_opt_buf = Aligned::<i8, MAX_SIZE>::new(0);
    let mut profile = ProfileSummary::new();

    for (iter, &size) in sizes.iter().enumerate() {
        let out_ansi = out_ansi_buf.slice_mut(size as usize);
        let out_opt = out_opt_buf.slice_mut(size as usize);
        for i in 0..size as usize {
            let v = rng.rand_i8_255();
            out_ansi[i] = v;
            out_opt[i] = v;
        }

        let t0 = ccount();
        unsafe {
            esp_nn_relu6_s8_ansi(out_ansi.as_mut_ptr(), size);
        }
        let c_ansi = ccount().wrapping_sub(t0);

        let t0 = ccount();
        unsafe {
            esp_nn_relu6_s8_esp32s3(out_opt.as_mut_ptr(), size);
        }
        let c_opt = ccount().wrapping_sub(t0);

        profile.record("relu6_s8", iter, c_ansi, c_opt, out_ansi, out_opt);
    }

    profile.print("relu6_s8");
}

fn test_hard_swish(rng: &mut NewlibRand) {
    const MAX_SIZE: usize = 12544;
    const MAX_SCRATCH_SIZE: usize = 1024;

    let sizes: &[i32] = &[1, 8, 16, 32, 100, 1024, 12544];
    let reluish_exps: &[i32] = &[2, -1, 0];
    let output_exps: &[i32] = &[-1, -2, -1];
    let input_zp: i16 = -128;
    let output_mult_fxp: i16 = 19661;
    let reluish_mult_fxp: i16 = 22938;
    let output_zp: i16 = -128;
    let mut scratch_buf = Aligned::<u8, MAX_SCRATCH_SIZE>::new(0);
    let mut input_buf = Aligned::<i8, MAX_SIZE>::new(0);
    let mut out_ansi_buf = Aligned::<i8, MAX_SIZE>::new(0);
    let mut out_opt_buf = Aligned::<i8, MAX_SIZE>::new(0);
    let mut profile = ProfileSummary::new();

    let scratch_size = unsafe { esp_nn_get_hard_swish_scratch_size_esp32s3() } as usize;
    let scratch = scratch_buf.slice_mut(scratch_size.max(1));
    unsafe {
        esp_nn_set_hard_swish_scratch_buf_esp32s3(scratch.as_mut_ptr() as *mut _);
    }

    for (iter, &size) in sizes.iter().enumerate() {
        let input = input_buf.slice_mut(size as usize);
        for v in input.iter_mut() {
            *v = rng.rand_i8();
        }
        for exp_idx in 0..3 {
            let out_ansi = out_ansi_buf.slice_mut(size as usize);
            let out_opt = out_opt_buf.slice_mut(size as usize);
            out_ansi.fill(0);
            out_opt.fill(0);

            let t0 = ccount();
            unsafe {
                esp_nn_hard_swish_s8_ansi(
                    input.as_ptr(),
                    out_ansi.as_mut_ptr(),
                    size,
                    input_zp,
                    output_mult_fxp,
                    reluish_mult_fxp,
                    reluish_exps[exp_idx],
                    output_exps[exp_idx],
                    output_zp,
                );
            }
            let c_ansi = ccount().wrapping_sub(t0);

            let t0 = ccount();
            unsafe {
                esp_nn_hard_swish_s8_esp32s3(
                    input.as_ptr(),
                    out_opt.as_mut_ptr(),
                    size,
                    input_zp,
                    output_mult_fxp,
                    reluish_mult_fxp,
                    reluish_exps[exp_idx],
                    output_exps[exp_idx],
                    output_zp,
                );
            }
            let c_opt = ccount().wrapping_sub(t0);

            profile.record(
                "hard_swish_s8",
                iter * 3 + exp_idx,
                c_ansi,
                c_opt,
                out_ansi,
                out_opt,
            );
        }
    }

    profile.print("hard_swish_s8");
}

fn test_mean(rng: &mut NewlibRand) {
    const MAX_INPUT_SIZE: usize = 14 * 14 * 120;
    const MAX_OUT_CH: usize = 576;

    let cases: &[(i32, i32, i32)] = &[
        (7, 7, 16),
        (7, 7, 72),
        (14, 14, 40),
        (14, 14, 120),
        (28, 28, 24),
        (1, 1, 576),
        (3, 3, 96),
    ];
    let input_zp = -128;
    let output_zp = -128;
    let multiplier = 1073741824;
    let shift = -1;
    let mut input_buf = Aligned::<i8, MAX_INPUT_SIZE>::new(0);
    let mut out_ansi_buf = Aligned::<i8, MAX_OUT_CH>::new(0);
    let mut out_opt_buf = Aligned::<i8, MAX_OUT_CH>::new(0);
    let mut profile = ProfileSummary::new();

    for (iter, &(h, w, ch)) in cases.iter().enumerate() {
        let input = input_buf.slice_mut((h * w * ch) as usize);
        let out_ansi = out_ansi_buf.slice_mut(ch as usize);
        let out_opt = out_opt_buf.slice_mut(ch as usize);
        out_ansi.fill(0);
        out_opt.fill(0);
        for v in input.iter_mut() {
            *v = rng.rand_i8();
        }

        let t0 = ccount();
        unsafe {
            esp_nn_mean_nhwc_s8_ansi(
                input.as_ptr(),
                out_ansi.as_mut_ptr(),
                h,
                w,
                ch,
                input_zp,
                output_zp,
                multiplier,
                shift,
            );
        }
        let c_ansi = ccount().wrapping_sub(t0);

        let t0 = ccount();
        unsafe {
            esp_nn_mean_nhwc_s8_esp32s3(
                input.as_ptr(),
                out_opt.as_mut_ptr(),
                h,
                w,
                ch,
                input_zp,
                output_zp,
                multiplier,
                shift,
            );
        }
        let c_opt = ccount().wrapping_sub(t0);

        profile.record("mean_nhwc_s8", iter, c_ansi, c_opt, out_ansi, out_opt);
    }

    profile.print("mean_nhwc_s8");
}

fn test_depthwise_conv(rng: &mut NewlibRand) {
    const MAX_INPUT_SIZE: usize = 28 * 28 * 64;
    const MAX_FILTER_SIZE: usize = 5 * 5 * 4 * 8 + 4;
    const MAX_OUT_SIZE: usize = 48 * 48 * 8;
    const MAX_BIAS_SIZE: usize = 64 + 1;
    const MAX_SCRATCH_SIZE: usize = 65536;

    let input_offset: i32 = 5;
    let out_offset: i32 = 7;
    let act_min: i32 = -125;
    let act_max: i32 = 120;

    let mut input_buf = Aligned::<i8, MAX_INPUT_SIZE>::new(0);
    let mut filter_buf = Aligned::<i8, MAX_FILTER_SIZE>::new(0);
    let mut bias_buf = Aligned::<i32, MAX_BIAS_SIZE>::new(0);
    let mut out_mult_buf = Aligned::<i32, MAX_BIAS_SIZE>::new(0);
    let mut out_shift_buf = Aligned::<i32, MAX_BIAS_SIZE>::new(0);
    let mut out_ansi_buf = Aligned::<i8, MAX_OUT_SIZE>::new(0);
    let mut out_opt_buf = Aligned::<i8, MAX_OUT_SIZE>::new(0);
    let mut scratch_buf = Aligned::<u8, MAX_SCRATCH_SIZE>::new(0);
    let mut profile = ProfileSummary::new();

    for iter in 0..17 {
        let (in_wd, in_ht, channels, ch_mult, f_wd, f_ht, pad, stride) = match iter {
            0 => (18, 18, 16, 1, 3, 3, 0, 1),
            1 => (10, 10, 16, 1, 3, 3, 1, 1),
            2 => (10, 10, 24, 1, 3, 3, 1, 1),
            3 => (10, 10, 24, 1, 3, 3, 1, 1),
            4 => (6, 6, 4, 8, 3, 3, 1, 1),
            5 => (12, 12, 4, 8, 5, 5, 1, 1),
            6 => (6, 6, 4, 4, 3, 3, 1, 1),
            7 => (6, 6, 16, 1, 3, 3, 0, 2),
            8 => (28, 28, 64, 1, 3, 3, 0, 2),
            9 => (6, 6, 16, 1, 3, 3, 0, 2),
            15 => (48, 48, 8, 1, 3, 3, 1, 1),
            16 => (12, 12, 8, 1, 3, 3, 0, 2),
            _ => {
                let stride = rng.rand() % 2 + 1;
                let pad = if stride == 1 { 0 } else { rng.rand() % 2 };
                (6, 6, 16, 1, 3, 3, pad, stride)
            }
        };
        let out_wd = if pad > 0 {
            (in_wd + stride - 1) / stride
        } else {
            (in_wd + stride - f_wd) / stride
        };
        let out_ht = if pad > 0 {
            (in_ht + stride - 1) / stride
        } else {
            (in_ht + stride - f_ht) / stride
        };

        let in_size = (in_wd * in_ht * channels) as usize;
        let filter_size = (f_wd * f_ht * channels * ch_mult) as usize + 4;
        let out_size = (out_wd * out_ht * channels * ch_mult) as usize;
        let bias_size = (channels * ch_mult) as usize + 1;
        let quant_size = (channels * ch_mult) as usize;

        let input = input_buf.slice_mut(in_size);
        let filter = filter_buf.slice_mut(filter_size);
        let bias = bias_buf.slice_mut(bias_size);
        let out_mult = out_mult_buf.slice_mut(quant_size);
        let out_shift = out_shift_buf.slice_mut(quant_size);
        let out_ansi = out_ansi_buf.slice_mut(out_size);
        let out_opt = out_opt_buf.slice_mut(out_size);
        out_ansi.fill(0);
        out_opt.fill(0);

        for v in input.iter_mut() {
            *v = rng.rand_i8_128();
        }
        for v in filter.iter_mut() {
            *v = rng.rand_i8();
        }
        for i in 0..quant_size {
            bias[i + 1] = rng.rand_i32_int16();
            out_shift[i] = -8 + rng.rand() % 3;
            out_mult[i] = 0x7eb0e200 + rng.rand() % 50;
        }

        let input_dims = data_dims_t {
            width: in_wd,
            height: in_ht,
            channels,
            extra: 1,
        };
        let filter_dims = data_dims_t {
            width: f_wd,
            height: f_ht,
            channels: 0,
            extra: 0,
        };
        let output_dims = data_dims_t {
            width: out_wd,
            height: out_ht,
            channels: channels * ch_mult,
            extra: 1,
        };
        let dw_params = dw_conv_params_t {
            in_offset: input_offset,
            out_offset,
            ch_mult,
            stride: data_2d_t {
                width: stride,
                height: stride,
            },
            padding: data_2d_t {
                width: pad,
                height: pad,
            },
            dilation: data_2d_t {
                width: 0,
                height: 0,
            },
            activation: act_params_t {
                min: act_min,
                max: act_max,
            },
        };
        let q_data = quant_data_t {
            shift: out_shift.as_mut_ptr(),
            mult: out_mult.as_mut_ptr(),
        };

        let t0 = ccount();
        unsafe {
            esp_nn_depthwise_conv_s8_ansi(
                &input_dims,
                input.as_ptr(),
                &filter_dims,
                filter.as_ptr().add(4),
                bias.as_ptr().add(1),
                &output_dims,
                out_ansi.as_mut_ptr(),
                &dw_params,
                &q_data,
            );
        }
        let c_ansi = ccount().wrapping_sub(t0);

        let scratch_size = unsafe {
            esp_nn_get_depthwise_conv_scratch_size_esp32s3(
                &input_dims,
                &filter_dims,
                &output_dims,
                &dw_params,
            )
        } as usize;
        if scratch_size > 0 {
            let scratch = scratch_buf.slice_mut(scratch_size);
            unsafe {
                esp_nn_set_depthwise_conv_scratch_buf_esp32s3(scratch.as_mut_ptr() as *mut _);
            }
        }

        let t0 = ccount();
        unsafe {
            esp_nn_depthwise_conv_s8_esp32s3(
                &input_dims,
                input.as_ptr(),
                &filter_dims,
                filter.as_ptr().add(4),
                bias.as_ptr().add(1),
                &output_dims,
                out_opt.as_mut_ptr(),
                &dw_params,
                &q_data,
            );
        }
        let c_opt = ccount().wrapping_sub(t0);

        profile.record("dw_conv_s8", iter, c_ansi, c_opt, out_ansi, out_opt);
    }

    profile.print("depthwise_conv_s8");
}

fn test_conv(rng: &mut NewlibRand) {
    const MAX_INPUT_SIZE: usize = 48 * 48 * 32;
    const MAX_FILTER_SIZE: usize = 3 * 3 * 12 * 64 + 16;
    const MAX_OUT_SIZE: usize = 48 * 48 * 32;
    const MAX_BIAS_SIZE: usize = 64;
    const MAX_SCRATCH_SIZE: usize = 65536;

    let mut input_buf = Aligned::<i8, MAX_INPUT_SIZE>::new(0);
    let mut filter_buf = Aligned::<i8, MAX_FILTER_SIZE>::new(0);
    let mut bias_buf = Aligned::<i32, MAX_BIAS_SIZE>::new(0);
    let mut out_mult_buf = Aligned::<i32, MAX_BIAS_SIZE>::new(0);
    let mut out_shift_buf = Aligned::<i32, MAX_BIAS_SIZE>::new(0);
    let mut out_ansi_buf = Aligned::<i8, MAX_OUT_SIZE>::new(0);
    let mut out_opt_buf = Aligned::<i8, MAX_OUT_SIZE>::new(0);
    let mut scratch_buf = Aligned::<u8, MAX_SCRATCH_SIZE>::new(0);
    let mut profile = ProfileSummary::new();

    for iter in 0..18 {
        let mut i_off = 5;
        let mut o_off = 3;
        let mut a_min = -125;
        let mut a_max = 122;
        let (in_wd, in_ht, in_ch, out_ch, f_wd, f_ht, pad, swd, sht) = match iter {
            0 => (10, 10, 64, 64, 1, 1, 0, 1, 1),
            1 => (4, 4, 20, 8, 1, 1, 0, 1, 1),
            2 => (10, 10, 3, 64, 3, 3, 0, 1, 1),
            3 => (10, 10, 3, 64, 1, 1, 0, 1, 1),
            4 => (10, 10, 12, 64, 3, 3, 1, 1, 1),
            5 => (16, 16, 16, 16, 1, 1, 0, 2, 2),
            6 => (2, 2, 8, 8, 1, 1, 0, 1, 1),
            7 => (112, 112, 3, 16, 6, 6, 0, 2, 2),
            8 => (8, 8, 5, 16, 6, 6, 0, 2, 2),
            9 => (3, 3, 32, 1, 3, 3, 1, 1, 1),
            10 => (4, 8, 1, 3, 3, 3, 0, 2, 2),
            11 => (4, 8, 3, 4, 3, 3, 0, 2, 2),
            15 => {
                i_off = 127;
                o_off = 39;
                (48, 48, 32, 32, 1, 1, 0, 1, 1)
            }
            16 => {
                i_off = 127;
                o_off = 39;
                a_min = -128;
                a_max = 127;
                (48, 48, 32, 32, 1, 1, 0, 1, 1)
            }
            17 => {
                i_off = 110;
                o_off = 39;
                a_min = -128;
                a_max = 127;
                (24, 24, 32, 32, 1, 1, 0, 1, 1)
            }
            _ => (8, 8, 16, 16, 1, 1, 0, 1, 1),
        };
        let out_wd = if pad > 0 {
            (in_wd + swd - 1) / swd
        } else {
            (in_wd + swd - f_wd) / swd
        };
        let out_ht = if pad > 0 {
            (in_ht + sht - 1) / sht
        } else {
            (in_ht + sht - f_ht) / sht
        };

        let in_size = (in_wd * in_ht * in_ch) as usize;
        let filter_offset = if iter == 17 { 5usize } else { 0usize };
        let filter_size = (f_wd * f_ht * in_ch * out_ch) as usize + 2;
        let out_size = (out_wd * out_ht * out_ch) as usize;
        let bias_size = out_ch as usize;

        let input = input_buf.slice_mut(in_size);
        let filter_storage = filter_buf.slice_mut(filter_size + filter_offset);
        let filter = &mut filter_storage[..filter_size];
        let bias = bias_buf.slice_mut(bias_size);
        let out_mult = out_mult_buf.slice_mut(bias_size);
        let out_shift = out_shift_buf.slice_mut(bias_size);
        let out_ansi = out_ansi_buf.slice_mut(out_size);
        let out_opt = out_opt_buf.slice_mut(out_size);
        out_ansi.fill(0);
        out_opt.fill(0);

        for v in input.iter_mut() {
            *v = rng.rand_i8_255();
        }
        for v in filter.iter_mut() {
            *v = rng.rand_i8();
        }
        for k in bias.iter_mut().take(bias_size) {
            *k = rng.rand_u16_plus_u8();
        }
        for i in 0..bias_size {
            out_shift[i] = rng.rand_shift_conv();
            out_mult[i] = rng.rand_mult_conv();
        }
        if iter == 17 {
            for shift in out_shift.iter_mut() {
                *shift = -6;
            }
        }
        if iter == 16 {
            input[..YOLO_INPUT.len()].copy_from_slice(&YOLO_INPUT);
            filter[..YOLO_FILTER.len()].copy_from_slice(&YOLO_FILTER);
            bias[..YOLO_BIAS.len()].copy_from_slice(&YOLO_BIAS);
            out_shift[..YOLO_SHIFTS.len()].copy_from_slice(&YOLO_SHIFTS);
            out_mult[..YOLO_MULTS.len()].copy_from_slice(&YOLO_MULTS);
        }

        let input_dims = data_dims_t {
            width: in_wd,
            height: in_ht,
            channels: in_ch,
            extra: 1,
        };
        let filter_dims = data_dims_t {
            width: f_wd,
            height: f_ht,
            channels: in_ch,
            extra: 1,
        };
        let output_dims = data_dims_t {
            width: out_wd,
            height: out_ht,
            channels: out_ch,
            extra: 1,
        };
        let conv_params = conv_params_t {
            in_offset: i_off,
            out_offset: o_off,
            stride: data_2d_t {
                width: swd,
                height: sht,
            },
            padding: data_2d_t {
                width: pad,
                height: pad,
            },
            dilation: data_2d_t {
                width: 0,
                height: 0,
            },
            activation: act_params_t {
                min: a_min,
                max: a_max,
            },
        };
        let q_data = quant_data_t {
            shift: out_shift.as_mut_ptr(),
            mult: out_mult.as_mut_ptr(),
        };

        let t0 = ccount();
        unsafe {
            esp_nn_conv_s8_ansi(
                &input_dims,
                input.as_ptr(),
                &filter_dims,
                filter.as_ptr().add(filter_offset),
                bias.as_ptr(),
                &output_dims,
                out_ansi.as_mut_ptr(),
                &conv_params,
                &q_data,
            );
        }
        let c_ansi = ccount().wrapping_sub(t0);

        let scratch_size = unsafe {
            esp_nn_get_conv_scratch_size_esp32s3(
                &input_dims,
                &filter_dims,
                &output_dims,
                &conv_params,
            )
        } as usize;
        if scratch_size > 0 {
            let scratch = scratch_buf.slice_mut(scratch_size);
            unsafe {
                esp_nn_set_conv_scratch_buf_esp32s3(scratch.as_mut_ptr() as *mut _);
            }
        }

        let t0 = ccount();
        unsafe {
            esp_nn_conv_s8_esp32s3(
                &input_dims,
                input.as_ptr(),
                &filter_dims,
                filter.as_ptr().add(filter_offset),
                bias.as_ptr(),
                &output_dims,
                out_opt.as_mut_ptr(),
                &conv_params,
                &q_data,
            );
        }
        let c_opt = ccount().wrapping_sub(t0);

        profile.record("conv_s8", iter, c_ansi, c_opt, out_ansi, out_opt);
    }

    profile.print("conv_s8");
}

#[main]
fn main() -> ! {
    rtt_target::rtt_init_defmt!();
    let config = esp_hal::Config::default().with_cpu_clock(CpuClock::max());
    let _peripherals = esp_hal::init(config);
    esp_alloc::heap_allocator!(#[esp_hal::ram(reclaimed)] size: 73744);
    let delay = Delay::new();
    let mut rng = NewlibRand::new();

    log_section("Running s8 tests...");
    test_add(&mut rng);
    test_mul(&mut rng);
    test_mul_broadcast_channel(&mut rng);
    test_depthwise_conv(&mut rng);
    test_conv(&mut rng);
    test_relu6(&mut rng);
    test_avg_pool(&mut rng);
    test_max_pool(&mut rng);
    test_fully_connected(&mut rng);
    test_fully_connected_per_ch(&mut rng);
    test_softmax(&mut rng);
    test_hard_swish(&mut rng);
    test_mean(&mut rng);
    log_section("s8 tests done!");

    loop {
        delay.delay_millis(50);
    }
}

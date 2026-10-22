//! int8 looped-transformer world model for the Pico 2 W. `no_std`, so the same code runs on
//! the host (tests, terminal demo) and on the RP2350.
#![cfg_attr(not(test), no_std)]

pub mod agent;
/// The exported agent this build runs (`TWF_AGENT`, see `build.rs`).
#[allow(clippy::all)]
#[rustfmt::skip]
pub mod agent_weights {
    include!(concat!(env!("TWF_AGENT_DIR"), "/agent_weights.rs"));
}
pub mod dream;
pub mod learner;
pub mod model;
/// The exported world model ("env") this build runs (`TWF_ENV`, see `build.rs`).
#[allow(clippy::all)]
#[rustfmt::skip]
pub mod weights {
    include!(concat!(env!("TWF_ENV_DIR"), "/weights.rs"));
}

/// Names of the env and agent this build was made with.
pub const ENV_NAME: &str = env!("TWF_ENV_NAME");
pub const AGENT_NAME: &str = env!("TWF_AGENT_NAME");

/// Portable model of `dot4`: bytes 0/2 and 1/3 are paired exactly like SXTB16 + SMLAD do.
pub fn dot4_ref(a: u32, w: u32, acc: i32) -> i32 {
    let b = |x: u32, i: u32| ((x >> (8 * i)) as u8 as i8) as i32;
    let even = b(a, 0) * b(w, 0) + b(a, 2) * b(w, 2);
    let odd = b(a, 1) * b(w, 1) + b(a, 3) * b(w, 3);
    acc.wrapping_add(even).wrapping_add(odd)
}

/// Same as `dot4_ref` with `a` holding unsigned bytes.
pub fn dot4_u8_ref(a: u32, w: u32, acc: i32) -> i32 {
    let u = |x: u32, i: u32| ((x >> (8 * i)) as u8) as i32;
    let s = |x: u32, i: u32| ((x >> (8 * i)) as u8 as i8) as i32;
    let even = u(a, 0) * s(w, 0) + u(a, 2) * s(w, 2);
    let odd = u(a, 1) * s(w, 1) + u(a, 3) * s(w, 3);
    acc.wrapping_add(even).wrapping_add(odd)
}

/// Four int8 MACs from packed words: SXTB16 unpacks two bytes into halfwords, SMLAD does 2 MACs.
/// With `UNSIGNED_A`, `a` is unpacked with UXTB16 instead (uint8 x int8).
#[cfg(target_arch = "arm")]
#[inline(always)]
fn dot4<const UNSIGNED_A: bool>(a: u32, w: u32, mut acc: i32) -> i32 {
    use core::arch::asm;
    let (a0, a1, w0, w1): (u32, u32, u32, u32);
    // SAFETY: pure register arithmetic on the DSP extension (enabled by target-cpu=cortex-m33).
    unsafe {
        if UNSIGNED_A {
            asm!("uxtb16 {o}, {i}", o = out(reg) a0, i = in(reg) a, options(pure, nomem, nostack));
            asm!("uxtb16 {o}, {i}, ror #8", o = out(reg) a1, i = in(reg) a, options(pure, nomem, nostack));
        } else {
            asm!("sxtb16 {o}, {i}", o = out(reg) a0, i = in(reg) a, options(pure, nomem, nostack));
            asm!("sxtb16 {o}, {i}, ror #8", o = out(reg) a1, i = in(reg) a, options(pure, nomem, nostack));
        }
        asm!("sxtb16 {o}, {i}", o = out(reg) w0, i = in(reg) w, options(pure, nomem, nostack));
        asm!("sxtb16 {o}, {i}, ror #8", o = out(reg) w1, i = in(reg) w, options(pure, nomem, nostack));
        asm!("smlad {acc}, {x}, {y}, {acc}", acc = inout(reg) acc, x = in(reg) a0, y = in(reg) w0, options(pure, nomem, nostack));
        asm!("smlad {acc}, {x}, {y}, {acc}", acc = inout(reg) acc, x = in(reg) a1, y = in(reg) w1, options(pure, nomem, nostack));
    }
    acc
}

#[cfg(not(target_arch = "arm"))]
#[inline(always)]
fn dot4<const UNSIGNED_A: bool>(a: u32, w: u32, acc: i32) -> i32 {
    if UNSIGNED_A { dot4_u8_ref(a, w, acc) } else { dot4_ref(a, w, acc) }
}

/// One little-endian 32-bit load of a 4-byte chunk (a single LDR; unaligned is fine on the M33).
#[inline(always)]
fn word<T: Copy>(c: &[T]) -> u32 {
    debug_assert!(c.len() == 4 && size_of::<T>() == 1);
    // SAFETY: `c` is a 4-element chunk of a 1-byte type, so 4 readable bytes.
    u32::from_le(unsafe { c.as_ptr().cast::<u32>().read_unaligned() })
}

/// Dot product of two int8 vectors whose length is a multiple of 4.
#[inline]
pub fn dot_i8(a: &[i8], w: &[i8], mut acc: i32) -> i32 {
    debug_assert!(a.len() == w.len() && a.len().is_multiple_of(4));
    for (ca, cw) in a.chunks_exact(4).zip(w.chunks_exact(4)) {
        acc = dot4::<false>(word(ca), word(cw), acc);
    }
    acc
}

/// Dot product of a uint8 vector with an int8 vector, length a multiple of 4.
#[inline]
pub fn dot_u8_i8(a: &[u8], w: &[i8], mut acc: i32) -> i32 {
    debug_assert!(a.len() == w.len() && a.len().is_multiple_of(4));
    for (ca, cw) in a.chunks_exact(4).zip(w.chunks_exact(4)) {
        acc = dot4::<true>(word(ca), word(cw), acc);
    }
    acc
}

#[cfg(test)]
mod tests {
    use super::*;

    pub(crate) struct Lcg(pub u64);
    impl Lcg {
        pub fn next(&mut self) -> u32 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (self.0 >> 33) as u32
        }
        fn i8s(&mut self, n: usize) -> Vec<i8> {
            (0..n).map(|_| self.next() as u8 as i8).collect()
        }
    }

    #[test]
    fn dot4_pairing_matches_naive_on_extremes() {
        let mut r = Lcg(1);
        let mut cases: Vec<(u32, u32)> = (0..20_000).map(|_| (r.next() ^ r.next() << 8, r.next() ^ r.next() << 8)).collect();
        cases.extend([(0x8080_8080, 0x8080_8080), (0x7f7f_7f7f, 0x8080_8080), (0, u32::MAX), (u32::MAX, u32::MAX)]);
        for (a, w) in cases {
            let (ab, wb) = (a.to_le_bytes(), w.to_le_bytes());
            let signed: i32 = ab.iter().zip(wb).map(|(&x, y)| x as i8 as i32 * y as i8 as i32).sum();
            let unsigned: i32 = ab.iter().zip(wb).map(|(&x, y)| x as i32 * y as i8 as i32).sum();
            assert_eq!(dot4_ref(a, w, 7), 7 + signed);
            assert_eq!(dot4_u8_ref(a, w, 7), 7 + unsigned);
        }
    }

    #[test]
    fn dots_match_naive() {
        let mut r = Lcg(2);
        for k in [4, 16, 32, 80, 1024] {
            let (a, w) = (r.i8s(k), r.i8s(k));
            let au: Vec<u8> = a.iter().map(|&x| x as u8).collect();
            let naive: i32 = a.iter().zip(&w).map(|(&x, &y)| x as i32 * y as i32).sum();
            let naive_u: i32 = au.iter().zip(&w).map(|(&x, &y)| x as i32 * y as i32).sum();
            assert_eq!(dot_i8(&a, &w, -5), naive - 5);
            assert_eq!(dot_u8_i8(&au, &w, -5), naive_u - 5);
        }
        let (a, w) = (vec![-128i8; 1024], vec![-128i8; 1024]);
        assert_eq!(dot_i8(&a, &w, 0), 16384 * 1024); // worst case still fits i32
    }
}

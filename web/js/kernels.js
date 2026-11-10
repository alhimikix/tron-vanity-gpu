// WGSL kernels for the browser grinder. Same scheme as the CUDA version:
//   start: k0 = ChaCha12(key, chunk) mod n -> k0*G over a 4-bit window table -> affine centre (x, y)
//   walk:  per round the centre P and P +/- j*G (j = 1..M) share one field inversion; P moves by STEP*G.
//          Every point: Keccak-256 -> 0x41 || h20 -> gates (top-bit ranges, checksum low byte) -> exact base58.
import { FE_WGSL } from "./fe.wgsl.js";
import { FOLD, RHO, PI, KRC, SK } from "./ref.js";

export const M = 32;              // j*G for j = 1..M per round
export const STEP = 2 * M + 1;    // keys per round
export const ROUNDS = 64;         // rounds per random start: 64 * 65 = 4160 keys
export const REC = 10;            // u32 per match record: thread, offset, pattern, address words[7]
export const PSTRIDE = 72;        // u32 per pattern: plen, slen, pre[34], suf[34], pad
export const SKIP_TAB = 272;      // skip buffer: byte table offset (after 128 targets)
export const SKIP_TARGETS = 128;

const hex = (v) => "0x" + (v >>> 0).toString(16) + "u";

// rotate a 64-bit lane held as vec2(lo, hi) left by n
function rolExpr(v, n) {
    if (n === 0) return v;
    if (n === 32) return `vec2<u32>(${v}.y, ${v}.x)`;
    const [a, b, m] = n < 32 ? ["x", "y", n] : ["y", "x", n - 32];
    return `vec2<u32>((${v}.${a} << ${m}u) | (${v}.${b} >> ${32 - m}u), (${v}.${b} << ${m}u) | (${v}.${a} >> ${32 - m}u))`;
}

function keccakWgsl() {
    const L = [];
    L.push("fn keccak_f(st: ptr<function, array<vec2<u32>, 25>>) {");
    for (let i = 0; i < 25; i++) L.push(`    var s${i} = (*st)[${i}];`);
    L.push("    for (var r = 0u; r < 24u; r++) {");
    for (let i = 0; i < 5; i++) L.push(`        let c${i} = s${i} ^ s${i + 5} ^ s${i + 10} ^ s${i + 15} ^ s${i + 20};`);
    for (let i = 0; i < 5; i++) {
        L.push(`        let d${i} = c${(i + 4) % 5} ^ ${rolExpr(`c${(i + 1) % 5}`, 1)};`);
        L.push(`        ${[0, 5, 10, 15, 20].map((j) => `s${j + i} ^= d${i};`).join(" ")}`);
    }
    L.push("        var t = s1; var u: vec2<u32>;");
    for (let i = 0; i < 24; i++) L.push(`        u = s${PI[i]}; s${PI[i]} = ${rolExpr("t", RHO[i])}; t = u;`);
    for (let j = 0; j < 25; j += 5) {
        L.push(`        { let a0 = s${j}; let a1 = s${j + 1}; let a2 = s${j + 2}; let a3 = s${j + 3}; let a4 = s${j + 4};`);
        L.push(`          s${j} = a0 ^ (~a1 & a2); s${j + 1} = a1 ^ (~a2 & a3); s${j + 2} = a2 ^ (~a3 & a4); s${j + 3} = a3 ^ (~a4 & a0); s${j + 4} = a4 ^ (~a0 & a1); }`);
    }
    L.push("        s0 ^= KRC[r];");
    L.push("    }");
    for (let i = 0; i < 25; i++) L.push(`    (*st)[${i}] = s${i};`);
    L.push("}");
    return L.join("\n");
}

const KRC_LIST = KRC.map((v) => `vec2<u32>(${hex(Number(v & 0xffffffffn))}, ${hex(Number(v >> 32n))})`).join(", ");
const SK_LIST = SK.map(hex).join(", ");

export function shader() {
    const fold = FOLD.map((v) => `${v}u`).join(", ");
    return /* wgsl */ `
${FE_WGSL}

fn fe_one() -> Fe { var r: Fe; r[0] = 1u; return r; }
fn fe_sqn(a: Fe, n: i32) -> Fe { var r = a; for (var i = 0; i < n; i++) { r = fe_sq(r); } return r; }

// a^(p-2), the libsecp256k1 addition chain
fn fe_invert(a: Fe) -> Fe {
    let x2 = fe_mul(fe_sq(a), a);
    let x3 = fe_mul(fe_sq(x2), a);
    let x6 = fe_mul(fe_sqn(x3, 3), x3);
    let x9 = fe_mul(fe_sqn(x6, 3), x3);
    let x11 = fe_mul(fe_sqn(x9, 2), x2);
    let x22 = fe_mul(fe_sqn(x11, 11), x11);
    let x44 = fe_mul(fe_sqn(x22, 22), x22);
    let x88 = fe_mul(fe_sqn(x44, 44), x44);
    let x176 = fe_mul(fe_sqn(x88, 88), x88);
    let x220 = fe_mul(fe_sqn(x176, 44), x44);
    let x223 = fe_mul(fe_sqn(x220, 3), x3);
    var t = fe_mul(fe_sqn(x223, 23), x22);
    t = fe_mul(fe_sqn(t, 5), a);
    t = fe_mul(fe_sqn(t, 3), x2);
    return fe_mul(fe_sqn(t, 2), a);
}

// fully reduced (< p) limbs
fn fe_canon(a: Fe) -> Fe {
    var r = fe_carry(a);
    for (var rep = 0; rep < 2; rep++) {
        let c0 = r[19] >> 9u; r[19] &= 0x1ffu; r[0] += 977u * c0; r[2] += c0 << 6u;
        var c: u32 = 0u;
        for (var i = 0; i < 19; i++) { r[i] += c; c = r[i] >> 13u; r[i] &= 0x1fffu; }
        r[19] += c;
    }
    var s = r; s[0] += 977u; s[2] += 64u;   // r + 2^256 - p
    var c: u32 = 0u;
    for (var i = 0; i < 19; i++) { s[i] += c; c = s[i] >> 13u; s[i] &= 0x1fffu; }
    s[19] += c;
    if ((s[19] >> 9u) != 0u) { s[19] &= 0x1ffu; return s; }
    return r;
}

struct Params {
    key: array<vec4<u32>, 2>,
    bitmap: array<vec4<u32>, 2>,
    base_lo: u32, base_hi: u32, nthreads: u32, rounds: u32,
    round0: u32, npat: u32, icase: u32, nranges: u32,
    match_cap: u32, need_ck: u32, dump: u32, skip: u32,
}
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> tab: array<u32>;       // walk points, then the window table; 40 limbs each
@group(0) @binding(2) var<storage, read_write> CX: array<u32>;  // centres, packed words, word-major
@group(0) @binding(3) var<storage, read_write> CY: array<u32>;
@group(0) @binding(4) var<storage, read> pats: array<u32>;
@group(0) @binding(5) var<storage, read> ranges: array<u32>;    // per range: lo_hi, lo_lo, hi_hi, hi_lo
@group(0) @binding(6) var<storage, read_write> outb: array<atomic<u32>>;   // [count, pad, pad, pad, records...]
@group(0) @binding(7) var<storage, read_write> dumpb: array<u32>;
@group(0) @binding(8) var<storage, read> skipb: array<u32>;     // m * 1,2,4,8,16; ntarget; 0x41 term; targets; byte table

const M: u32 = ${M}u;
const STEP: u32 = ${STEP}u;
const WTAB: u32 = ${(M + 1) * 40}u;     // window table offset in tab
const FOLD = array<u32, 58>(${fold});
const KRC = array<vec2<u32>, 24>(${KRC_LIST});
const SK = array<u32, 64>(${SK_LIST});

fn tab_fe(off: u32) -> Fe { var r: Fe; for (var i = 0u; i < 20u; i++) { r[i] = tab[off + i]; } return r; }
fn bswap(x: u32) -> u32 { return ((x & 0xffu) << 24u) | ((x & 0xff00u) << 8u) | ((x >> 8u) & 0xff00u) | (x >> 24u); }
fn rotl(x: u32, n: u32) -> u32 { return (x << n) | (x >> (32u - n)); }
fn ror(x: u32, n: u32) -> u32 { return (x >> n) | (x << (32u - n)); }

// ---------------- ChaCha12 start scalar ----------------
fn qr(x: ptr<function, array<u32, 16>>, a: u32, b: u32, c: u32, d: u32) {
    (*x)[a] += (*x)[b]; (*x)[d] = rotl((*x)[d] ^ (*x)[a], 16u);
    (*x)[c] += (*x)[d]; (*x)[b] = rotl((*x)[b] ^ (*x)[c], 12u);
    (*x)[a] += (*x)[b]; (*x)[d] = rotl((*x)[d] ^ (*x)[a], 8u);
    (*x)[c] += (*x)[d]; (*x)[b] = rotl((*x)[b] ^ (*x)[c], 7u);
}
fn start_scalar(lo: u32, hi: u32) -> array<u32, 8> {
    var x = array<u32, 16>(0x61707865u, 0x3320646eu, 0x79622d32u, 0x6b206574u,
        P.key[0].x, P.key[0].y, P.key[0].z, P.key[0].w, P.key[1].x, P.key[1].y, P.key[1].z, P.key[1].w,
        lo, hi, 0x74726f6eu, 0x76616e74u);
    let j = x;
    for (var r = 0; r < 6; r++) {
        qr(&x, 0u, 4u, 8u, 12u); qr(&x, 1u, 5u, 9u, 13u); qr(&x, 2u, 6u, 10u, 14u); qr(&x, 3u, 7u, 11u, 15u);
        qr(&x, 0u, 5u, 10u, 15u); qr(&x, 1u, 6u, 11u, 12u); qr(&x, 2u, 7u, 8u, 13u); qr(&x, 3u, 4u, 9u, 14u);
    }
    var w: array<u32, 8>;
    for (var i = 0; i < 8; i++) { w[i] = x[i] + j[i]; }
    // reduce below n (one subtraction suffices)
    let n = array<u32, 8>(0xD0364141u, 0xBFD25E8Cu, 0xAF48A03Bu, 0xBAAEDCE6u, 0xFFFFFFFEu, 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu);
    var d: array<u32, 8>; var br = 0u;
    for (var i = 0; i < 8; i++) {
        let t = w[i] - n[i];
        let b2 = select(0u, 1u, w[i] < n[i]) | select(0u, 1u, t < br);
        d[i] = t - br; br = b2;
    }
    if (br == 0u) { return d; }
    return w;
}

// P (Jacobian) += Q (affine); P must not be infinity and P != +-Q
fn jac_madd(X: ptr<function, Fe>, Y: ptr<function, Fe>, Z: ptr<function, Fe>, qx: Fe, qy: Fe) {
    let z2 = fe_sq(*Z);
    let u2 = fe_mul(qx, z2);
    let s2 = fe_mul(fe_mul(qy, z2), *Z);
    let h = fe_sub(u2, *X);
    let r = fe_sub(s2, *Y);
    let h2 = fe_sq(h); let h3 = fe_mul(h, h2);
    let v = fe_mul(*X, h2);
    let x3 = fe_sub(fe_sub(fe_sub(fe_sq(r), h3), v), v);
    *Y = fe_sub(fe_mul(r, fe_sub(v, x3)), fe_mul(*Y, h3));
    *X = x3;
    *Z = fe_mul(*Z, h);
}

fn store_centre(t: u32, x: Fe, y: Fe) {
    let wx = fe_pack(fe_canon(x)); let wy = fe_pack(fe_canon(y));
    for (var l = 0u; l < 8u; l++) { CX[l * P.nthreads + t] = wx[l]; CY[l * P.nthreads + t] = wy[l]; }
}

@compute @workgroup_size(64)
fn start_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let t = gid.x;
    if (t >= P.nthreads) { return; }
    let lo = P.base_lo + t;
    let hi = P.base_hi + select(0u, 1u, lo < P.base_lo);
    let k = start_scalar(lo, hi);
    var X: Fe; var Y: Fe; var Z: Fe;
    var inf = true;
    for (var w = 0u; w < 64u; w++) {
        let d = (k[w >> 3u] >> ((w & 7u) * 4u)) & 15u;
        if (d == 0u) { continue; }
        let e = WTAB + (w * 15u + d - 1u) * 40u;
        if (inf) { X = tab_fe(e); Y = tab_fe(e + 20u); Z = fe_one(); inf = false; }
        else { jac_madd(&X, &Y, &Z, tab_fe(e), tab_fe(e + 20u)); }
    }
    let zi = fe_invert(Z);
    let zi2 = fe_sq(zi);
    store_centre(t, fe_mul(X, zi2), fe_mul(Y, fe_mul(zi2, zi)));
}

// ---------------- address and matching ----------------
${keccakWgsl()}

fn sha256_block(m: array<u32, 16>) -> array<u32, 8> {
    var w = m;
    var a = 0x6a09e667u; var b = 0xbb67ae85u; var c = 0x3c6ef372u; var d = 0xa54ff53au;
    var e = 0x510e527fu; var f = 0x9b05688cu; var g = 0x1f83d9abu; var h = 0x5be0cd19u;
    for (var i = 0u; i < 64u; i++) {
        if (i >= 16u) {
            let w15 = w[(i - 15u) & 15u]; let w2 = w[(i - 2u) & 15u];
            w[i & 15u] += (ror(w15, 7u) ^ ror(w15, 18u) ^ (w15 >> 3u)) + w[(i - 7u) & 15u] + (ror(w2, 17u) ^ ror(w2, 19u) ^ (w2 >> 10u));
        }
        let t1 = h + (ror(e, 6u) ^ ror(e, 11u) ^ ror(e, 25u)) + ((e & f) ^ (~e & g)) + SK[i] + w[i & 15u];
        let t2 = (ror(a, 2u) ^ ror(a, 13u) ^ ror(a, 22u)) + ((a & b) ^ (a & c) ^ (b & c));
        h = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
    }
    return array<u32, 8>(a + 0x6a09e667u, b + 0xbb67ae85u, c + 0x3c6ef372u, d + 0xa54ff53au,
                         e + 0x510e527fu, f + 0x9b05688cu, g + 0x1f83d9abu, h + 0x5be0cd19u);
}

// first 4 bytes of sha256d(0x41 || hash20), hb = hash bytes as big-endian words
fn checksum_of(hb: array<u32, 5>) -> u32 {
    var m: array<u32, 16>;
    m[0] = 0x41000000u | (hb[0] >> 8u);
    for (var i = 1; i < 5; i++) { m[i] = (hb[i - 1] << 24u) | (hb[i] >> 8u); }
    m[5] = (hb[4] << 24u) | 0x00800000u;
    m[15] = 168u;
    let d1 = sha256_block(m);
    var m2: array<u32, 16>;
    for (var i = 0; i < 8; i++) { m2[i] = d1[i]; }
    m2[8] = 0x80000000u;
    m2[15] = 256u;
    return sha256_block(m2)[0];
}

// exact base58 of the 25-byte address and the patterns; records a match
fn full_check(hb: array<u32, 5>, ck: u32, t: u32, off: i32) {
    var n: array<u32, 13>;   // 26 bytes (one leading zero) as big-endian 16-bit limbs
    n[0] = 0x41u;
    for (var i = 0; i < 5; i++) { n[1 + 2 * i] = hb[i] >> 16u; n[2 + 2 * i] = hb[i] & 0xffffu; }
    n[11] = ck >> 16u; n[12] = ck & 0xffffu;
    var dig: array<u32, 34>;   // least significant first; always 34 digits
    var start = 0u;
    for (var k = 0u; k < 34u; k++) {
        while (start < 13u && n[start] == 0u) { start++; }
        var rem = 0u;
        for (var j = start; j < 13u; j++) {
            let cur = (rem << 16u) | n[j];
            n[j] = cur / 58u; rem = cur % 58u;
        }
        dig[k] = rem;
    }
    for (var p = 0u; p < P.npat; p++) {
        let base = p * ${PSTRIDE}u;
        let plen = pats[base]; let slen = pats[base + 1u];
        var ok = true;
        for (var q = 0u; q < plen && ok; q++) {
            var c = dig[33u - q];
            if (P.icase != 0u) { c = FOLD[c]; }
            ok = c == pats[base + 2u + q];
        }
        for (var q = 0u; q < slen && ok; q++) {
            var c = dig[slen - 1u - q];
            if (P.icase != 0u) { c = FOLD[c]; }
            ok = c == pats[base + 36u + q];
        }
        if (ok) {
            let slot = atomicAdd(&outb[0], 1u);
            if (slot < P.match_cap) {
                let o = 4u + slot * ${REC}u;
                atomicStore(&outb[o], t);
                atomicStore(&outb[o + 1u], bitcast<u32>(off));
                atomicStore(&outb[o + 2u], p);
                atomicStore(&outb[o + 3u], 0x41u);
                for (var q = 0u; q < 5u; q++) { atomicStore(&outb[o + 4u + q], hb[q]); }
                atomicStore(&outb[o + 9u], ck);
            }
            return;
        }
    }
}

// Checksum skip for suffixes of one length K >= 6: N = A * 2^32 + ck, so the suffix fixes
// ck = (t - A * 2^32) mod 58^K, which is usually >= 2^32: then no checksum can match and the double
// SHA-256 is not needed. r = (A * 2^32) mod m from a per-byte table (20 x 256 residues, each < m < 2^59,
// so the sum of 21 terms fits in 64 bits); 64-bit values are vec2(lo, hi).
fn sk64(i: u32) -> vec2<u32> { return vec2<u32>(skipb[i], skipb[i + 1u]); }
fn add64(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> { let lo = a.x + b.x; return vec2<u32>(lo, a.y + b.y + select(0u, 1u, lo < a.x)); }
fn sub64(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> { return vec2<u32>(a.x - b.x, a.y - b.y - select(0u, 1u, a.x < b.x)); }
fn ge64(a: vec2<u32>, b: vec2<u32>) -> bool { return a.y > b.y || (a.y == b.y && a.x >= b.x); }
fn skip_residue(hb: array<u32, 5>) -> vec2<u32> {
    var acc = sk64(12u);   // 0x41 * 2^192 mod m
    for (var i = 0u; i < 20u; i++) {
        let b = (hb[i >> 2u] >> (24u - 8u * (i & 3u))) & 0xffu;
        acc = add64(acc, sk64(${SKIP_TAB}u + (i * 256u + b) * 2u));
    }
    for (var s = 4; s >= 0; s--) {   // acc < 21 m: subtract 16m, 8m, 4m, 2m, m
        let ms = sk64(u32(s) * 2u);
        if (ge64(acc, ms)) { acc = sub64(acc, ms); }
    }
    return acc;
}
// does some suffix value accept a checksum below 2^32? (ck = 0xffffffff: any; else exactly ck)
fn skip_wanted(r: vec2<u32>, ck: u32, any: bool) -> bool {
    let m = sk64(0u);
    var want = false;
    for (var v = 0u; v < skipb[10]; v++) {
        let tv = sk64(16u + 2u * v);
        var d: vec2<u32>;
        if (ge64(tv, r)) { d = sub64(tv, r); } else { d = sub64(add64(tv, m), r); }
        want = want || (d.y == 0u && (any || d.x == ck));
    }
    return want;
}

// Keys that still need the checksum in skip mode wait in a per-thread queue (hash words, offset, residue)
// and are hashed at the end of the round, so a SIMD group runs ~max(queue) hashes per round, not one per key.
const QW: u32 = 8u;
alias Queue = array<u32, ${STEP * 8}>;

fn check_point(x0: Fe, y0: Fe, t: u32, off: i32, di: u32, q: ptr<function, Queue>, qn: ptr<function, u32>) {
    let x = fe_pack(fe_canon(x0)); let y = fe_pack(fe_canon(y0));
    var st: array<vec2<u32>, 25>;
    for (var i = 0u; i < 4u; i++) {
        st[i] = vec2<u32>(bswap(x[7u - 2u * i]), bswap(x[6u - 2u * i]));
        st[4u + i] = vec2<u32>(bswap(y[7u - 2u * i]), bswap(y[6u - 2u * i]));
    }
    st[8] = vec2<u32>(1u, 0u);
    st[16] = vec2<u32>(0u, 0x80000000u);
    keccak_f(&st);
    let hb = array<u32, 5>(bswap(st[1].y), bswap(st[2].x), bswap(st[2].y), bswap(st[3].x), bswap(st[3].y));
    let thi = 0x41000000u | (hb[0] >> 8u);
    let tlo = (hb[0] << 24u) | (hb[1] >> 8u);
    var hit = false;
    for (var r = 0u; r < P.nranges; r++) {
        let lh = ranges[r * 4u]; let ll = ranges[r * 4u + 1u]; let hh = ranges[r * 4u + 2u]; let hl = ranges[r * 4u + 3u];
        if ((thi > lh || (thi == lh && tlo >= ll)) && (thi < hh || (thi == hh && tlo <= hl))) { hit = true; break; }
    }
    var ck = 0u; var have = false;
    var res = vec2<u32>(0u, 0u);
    if (P.skip != 0u) {
        res = skip_residue(hb);
        if (P.dump != 0u) { ck = checksum_of(hb); have = true; hit = hit || skip_wanted(res, ck, false); }
        else if (!hit && skip_wanted(res, 0u, true)) {
            let e = *qn * QW;
            for (var i = 0u; i < 5u; i++) { (*q)[e + i] = hb[i]; }
            (*q)[e + 5u] = bitcast<u32>(off); (*q)[e + 6u] = res.x; (*q)[e + 7u] = res.y;
            *qn += 1u;
            return;
        }
    } else if (P.need_ck != 0u || P.dump != 0u) {
        ck = checksum_of(hb); have = true;
        let low = ck & 0xffu;
        hit = hit || (P.need_ck != 0u && ((P.bitmap[low >> 7u][(low >> 5u) & 3u] >> (low & 31u)) & 1u) != 0u);
    }
    if (P.dump != 0u) {   // selftest: address and skip residue of every key
        dumpb[di * 9u] = 0x41u;
        for (var q = 0u; q < 5u; q++) { dumpb[di * 9u + 1u + q] = hb[q]; }
        dumpb[di * 9u + 6u] = ck;
        dumpb[di * 9u + 7u] = res.x; dumpb[di * 9u + 8u] = res.y;
    }
    if (!hit) { return; }
    if (!have) { ck = checksum_of(hb); }
    full_check(hb, ck, t, off);
}

fn queue_flush(t: u32, q: ptr<function, Queue>, qn: ptr<function, u32>) {
    for (var k = 0u; k < *qn; k++) {
        let e = k * QW;
        let hb = array<u32, 5>((*q)[e], (*q)[e + 1u], (*q)[e + 2u], (*q)[e + 3u], (*q)[e + 4u]);
        let ck = checksum_of(hb);
        if (skip_wanted(vec2<u32>((*q)[e + 6u], (*q)[e + 7u]), ck, false)) { full_check(hb, ck, t, bitcast<i32>((*q)[e + 5u])); }
    }
    *qn = 0u;
}

// ---------------- walk ----------------
@compute @workgroup_size(64)
fn walk_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let t = gid.x;
    if (t >= P.nthreads) { return; }
    var wx: array<u32, 8>; var wy: array<u32, 8>;
    for (var l = 0u; l < 8u; l++) { wx[l] = CX[l * P.nthreads + t]; wy[l] = CY[l * P.nthreads + t]; }
    var x = fe_unpack(wx); var y = fe_unpack(wy);
    var pre: array<u32, ${(M + 1) * 10}>;   // prefix products, two 16-bit limbs per word
    var q: Queue; var qn = 0u;
    for (var r = 0u; r < P.rounds; r++) {
        let boff = i32((P.round0 + r) * STEP);
        let dbase = (t * P.rounds + r) * STEP + M;
        check_point(x, y, t, boff, dbase, &q, &qn);
        var acc: Fe;
        for (var j = 0u; j <= M; j++) {
            let dx = fe_sub(tab_fe(j * 40u), x);
            if (j == 0u) { acc = dx; } else { acc = fe_mul(acc, dx); }
            for (var i = 0u; i < 10u; i++) { pre[j * 10u + i] = acc[2u * i] | (acc[2u * i + 1u] << 16u); }
        }
        var inv = fe_invert(acc);
        var nx: Fe; var ny: Fe;
        for (var jj = i32(M); jj >= 0; jj--) {
            let j = u32(jj);
            var ij: Fe;
            if (j > 0u) {
                var pv: Fe;
                for (var i = 0u; i < 10u; i++) { let v = pre[(j - 1u) * 10u + i]; pv[2u * i] = v & 0xffffu; pv[2u * i + 1u] = v >> 16u; }
                ij = fe_mul(inv, pv);
            } else { ij = inv; }
            let gx = tab_fe(j * 40u); let gy = tab_fe(j * 40u + 20u);
            inv = fe_mul(inv, fe_sub(gx, x));
            // P + (j+1)G
            var l = fe_mul(fe_sub(gy, y), ij);
            var x3 = fe_sub(fe_sub(fe_sq(l), x), gx);
            var y3 = fe_sub(fe_mul(l, fe_sub(x, x3)), y);
            if (j == M) { nx = x3; ny = y3; continue; }   // next centre = P + STEP*G
            check_point(x3, y3, t, boff + i32(j) + 1, dbase + j + 1u, &q, &qn);
            // P - (j+1)G: slope -(gy + y) * ij
            l = fe_mul(fe_add(gy, y), ij);
            x3 = fe_sub(fe_sub(fe_sq(l), x), gx);
            y3 = fe_sub(fe_mul(l, fe_sub(x3, x)), y);
            check_point(x3, y3, t, boff - i32(j) - 1, dbase - j - 1u, &q, &qn);
        }
        if (qn != 0u) { queue_flush(t, &q, &qn); }
        x = nx; y = ny;
    }
    store_centre(t, x, y);
}
`;
}

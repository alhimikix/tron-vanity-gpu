// Reference secp256k1 (BigInt), Keccak-256, SHA-256, base58, ChaCha12 start scalars and pattern maths.
// Used to precompute curve points for the GPU and to re-check every GPU match.

export const P = (1n << 256n) - (1n << 32n) - 977n;
export const N = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141n;
export const G = [0x79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798n,
                  0x483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8n];
const mod = (a, m = P) => { a %= m; return a < 0n ? a + m : a; };

export function inv(a, m = P) {         // extended Euclid
    let r0 = mod(a, m), r1 = m, s0 = 1n, s1 = 0n;
    while (r1) { const q = r0 / r1; [r0, r1] = [r1, r0 - q * r1]; [s0, s1] = [s1, s0 - q * s1]; }
    return mod(s0, m);
}

// Jacobian points [X, Y, Z]; Z = 0 is infinity
const INF = [1n, 1n, 0n];
function jdbl([X, Y, Z]) {
    if (!Z || !Y) return INF;
    const yy = Y * Y % P, s = 4n * X * yy % P, m = 3n * X * X % P;
    const x3 = mod(m * m - 2n * s);
    return [x3, mod(m * (s - x3) - 8n * yy * yy), 2n * Y * Z % P];
}
function jadd(p, q) {
    if (!p[2]) return q;
    if (!q[2]) return p;
    const z1z1 = p[2] * p[2] % P, z2z2 = q[2] * q[2] % P;
    const u1 = p[0] * z2z2 % P, u2 = q[0] * z1z1 % P;
    const s1 = p[1] * q[2] % P * z2z2 % P, s2 = q[1] * p[2] % P * z1z1 % P;
    const h = mod(u2 - u1), r = mod(s2 - s1);
    if (!h) return r ? INF : jdbl(p);
    const hh = h * h % P, hhh = h * hh % P, v = u1 * hh % P;
    const x3 = mod(r * r - hhh - 2n * v);
    return [x3, mod(r * (v - x3) - s1 * hhh), p[2] * q[2] % P * h % P];
}
export function affine(p) {
    const zi = inv(p[2]), zi2 = zi * zi % P;
    return [p[0] * zi2 % P, p[1] * zi2 % P * zi % P];
}
// k * Q, plain double-and-add (no tables)
export function mulPoint(k, Q = G) {
    let r = INF, q = [Q[0], Q[1], 1n];
    k = mod(k, N);
    while (k) { if (k & 1n) r = jadd(r, q); q = jdbl(q); k >>= 1n; }
    return r;
}
export const mulG = (k) => affine(mulPoint(k));

// ---- Keccak-256 (original padding 0x01), SHA-256 ----
const RC = [0x0000000000000001n, 0x0000000000008082n, 0x800000000000808an, 0x8000000080008000n, 0x000000000000808bn,
    0x0000000080000001n, 0x8000000080008081n, 0x8000000000008009n, 0x000000000000008an, 0x0000000000000088n,
    0x0000000080008009n, 0x000000008000000an, 0x000000008000808bn, 0x800000000000008bn, 0x8000000000008089n,
    0x8000000000008003n, 0x8000000000008002n, 0x8000000000000080n, 0x000000000000800an, 0x800000008000000an,
    0x8000000080008081n, 0x8000000000008080n, 0x0000000080000001n, 0x8000000080008008n];
export const RHO = [1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 2, 14, 27, 41, 56, 8, 25, 43, 62, 18, 39, 61, 20, 44];
export const PI = [10, 7, 11, 17, 18, 3, 5, 16, 8, 21, 24, 4, 15, 23, 19, 13, 12, 2, 20, 14, 22, 9, 6, 1];
export const KRC = RC;
const M64 = (1n << 64n) - 1n;
const rol = (x, n) => ((x << BigInt(n)) | (x >> BigInt(64 - n))) & M64;
export function keccak256(bytes) {      // bytes.length < 136
    const s = new Array(25).fill(0n);
    const blk = new Uint8Array(136);
    blk.set(bytes); blk[bytes.length] ^= 0x01; blk[135] ^= 0x80;
    for (let i = 0; i < 17; i++) { let v = 0n; for (let b = 7; b >= 0; b--) v = (v << 8n) | BigInt(blk[8 * i + b]); s[i] ^= v; }
    for (let r = 0; r < 24; r++) {
        const c = [0, 1, 2, 3, 4].map((i) => s[i] ^ s[i + 5] ^ s[i + 10] ^ s[i + 15] ^ s[i + 20]);
        for (let i = 0; i < 5; i++) { const d = c[(i + 4) % 5] ^ rol(c[(i + 1) % 5], 1); for (let j = 0; j < 25; j += 5) s[j + i] ^= d; }
        let t = s[1];
        for (let i = 0; i < 24; i++) { const u = s[PI[i]]; s[PI[i]] = rol(t, RHO[i]); t = u; }
        for (let j = 0; j < 25; j += 5) {
            const a = s.slice(j, j + 5);
            for (let i = 0; i < 5; i++) s[j + i] = a[i] ^ (~a[(i + 1) % 5] & M64 & a[(i + 2) % 5]);
        }
        s[0] ^= RC[r];
    }
    const out = new Uint8Array(32);
    for (let i = 0; i < 32; i++) out[i] = Number((s[i >> 3] >> BigInt(8 * (i & 7))) & 0xffn);
    return out;
}

export const SK = [0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2];
export function sha256(bytes) {
    const len = bytes.length, nb = Math.ceil((len + 9) / 64);
    const m = new Uint8Array(nb * 64);
    m.set(bytes); m[len] = 0x80;
    const bits = len * 8;
    m[nb * 64 - 4] = bits >>> 24; m[nb * 64 - 3] = (bits >>> 16) & 255; m[nb * 64 - 2] = (bits >>> 8) & 255; m[nb * 64 - 1] = bits & 255;
    const h = [0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19];
    const ror = (x, n) => (x >>> n) | (x << (32 - n));
    const w = new Uint32Array(64);
    for (let b = 0; b < nb; b++) {
        for (let i = 0; i < 16; i++) { const o = b * 64 + 4 * i; w[i] = (m[o] << 24) | (m[o + 1] << 16) | (m[o + 2] << 8) | m[o + 3]; }
        for (let i = 16; i < 64; i++) {
            const a = w[i - 15], c = w[i - 2];
            w[i] = w[i - 16] + (ror(a, 7) ^ ror(a, 18) ^ (a >>> 3)) + w[i - 7] + (ror(c, 17) ^ ror(c, 19) ^ (c >>> 10));
        }
        let [A, B, C, D, E, F, G_, H] = h;
        for (let i = 0; i < 64; i++) {
            const t1 = (H + (ror(E, 6) ^ ror(E, 11) ^ ror(E, 25)) + ((E & F) ^ (~E & G_)) + SK[i] + w[i]) | 0;
            const t2 = ((ror(A, 2) ^ ror(A, 13) ^ ror(A, 22)) + ((A & B) ^ (A & C) ^ (B & C))) | 0;
            H = G_; G_ = F; F = E; E = (D + t1) | 0; D = C; C = B; B = A; A = (t1 + t2) | 0;
        }
        [A, B, C, D, E, F, G_, H].forEach((v, i) => { h[i] = (h[i] + v) | 0; });
    }
    const out = new Uint8Array(32);
    h.forEach((v, i) => { out[4 * i] = v >>> 24; out[4 * i + 1] = (v >>> 16) & 255; out[4 * i + 2] = (v >>> 8) & 255; out[4 * i + 3] = v & 255; });
    return out;
}

// ---- base58 ----
export const B58 = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";
export function b58encode(bytes) {
    let n = 0n;
    for (const b of bytes) n = (n << 8n) | BigInt(b);
    let s = "";
    while (n > 0n) { s = B58[Number(n % 58n)] + s; n /= 58n; }
    for (const b of bytes) { if (b !== 0) break; s = "1" + s; }
    return s;
}

// ---- Tron address: 0x41 || Keccak-256(X || Y)[12..32] || sha256d[0..4] ----
export function bigToBytesBE(x, n = 32) {
    const out = new Uint8Array(n);
    for (let i = n - 1; i >= 0; i--) { out[i] = Number(x & 0xffn); x >>= 8n; }
    return out;
}
export function addressBytes([x, y]) {
    const xy = new Uint8Array(64);
    xy.set(bigToBytesBE(x), 0); xy.set(bigToBytesBE(y), 32);
    const a = new Uint8Array(25);
    a[0] = 0x41; a.set(keccak256(xy).slice(12), 1);
    a.set(sha256(sha256(a.slice(0, 21))).slice(0, 4), 21);
    return a;
}
export const addressOfKey = (k) => b58encode(addressBytes(mulG(k)));
export const keyHex = (k) => k.toString(16).padStart(64, "0");

export function bigToWords(x) {         // 8 little-endian u32
    const w = new Uint32Array(8);
    for (let i = 0; i < 8; i++) { w[i] = Number(x & 0xffffffffn); x >>= 32n; }
    return w;
}
const wordsToBig = (w) => { let x = 0n; for (let i = 7; i >= 0; i--) x = (x << 32n) | BigInt(w[i] >>> 0); return x; };

// ---- ChaCha12: start scalar of chunk c = first 32 bytes of ChaCha12(key, counter = c), reduced below n ----
const rotl = (x, n) => ((x << n) | (x >>> (32 - n))) >>> 0;
export function chacha12Words(key, counter) {   // key: Uint32Array(8), counter: BigInt
    const x = new Uint32Array(16);
    x.set([0x61707865, 0x3320646e, 0x79622d32, 0x6b206574], 0); x.set(key, 4);
    x[12] = Number(counter & 0xffffffffn); x[13] = Number((counter >> 32n) & 0xffffffffn);
    x[14] = 0x74726f6e; x[15] = 0x76616e74;
    const j = x.slice();
    const qr = (a, b, c, d) => {
        x[a] = (x[a] + x[b]) >>> 0; x[d] = rotl(x[d] ^ x[a], 16);
        x[c] = (x[c] + x[d]) >>> 0; x[b] = rotl(x[b] ^ x[c], 12);
        x[a] = (x[a] + x[b]) >>> 0; x[d] = rotl(x[d] ^ x[a], 8);
        x[c] = (x[c] + x[d]) >>> 0; x[b] = rotl(x[b] ^ x[c], 7);
    };
    for (let r = 0; r < 6; r++) {
        qr(0, 4, 8, 12); qr(1, 5, 9, 13); qr(2, 6, 10, 14); qr(3, 7, 11, 15);
        qr(0, 5, 10, 15); qr(1, 6, 11, 12); qr(2, 7, 8, 13); qr(3, 4, 9, 14);
    }
    const out = new Uint32Array(8);
    for (let i = 0; i < 8; i++) out[i] = (x[i] + j[i]) >>> 0;
    return out;
}
export function startScalar(key, counter) {
    const w = wordsToBig(chacha12Words(key, counter));
    return w >= N ? w - N : w;
}
// private key of chunk `counter`, offset `off` from its start
export const keyOf = (key, counter, off) => mod(startScalar(key, counter) + BigInt(off), N);

// ---- GPU tables ----
// walk: j*G for j = 1..m, then step*G; start: d * 16^w * G for w = 0..63, d = 1..15
export function walkTable(m, step) {
    const pts = [];
    for (let j = 1; j <= m; j++) pts.push(mulG(BigInt(j)));
    pts.push(mulG(BigInt(step)));
    return pts;
}
export function windowTable() {
    const pts = [];
    let base = [G[0], G[1], 1n];
    for (let w = 0; w < 64; w++) {
        let acc = base;
        const row = [acc];
        for (let d = 2; d <= 15; d++) { acc = jadd(acc, base); row.push(acc); }
        // batch-normalise the row (one inversion per 15 points)
        const pre = [row[0][2]];
        for (let i = 1; i < 15; i++) pre.push(pre[i - 1] * row[i][2] % P);
        let iv = inv(pre[14]);
        const out = new Array(15);
        for (let i = 14; i >= 0; i--) {
            const zi = i ? iv * pre[i - 1] % P : iv;
            iv = iv * row[i][2] % P;
            const zi2 = zi * zi % P;
            out[i] = [row[i][0] * zi2 % P, row[i][1] * zi2 % P * zi % P];
        }
        pts.push(...out);
        base = jadd(row[14], base);   // 16 * base
    }
    return pts;
}
export function pointsToWords(pts) {
    const w = new Uint32Array(pts.length * 16);
    pts.forEach(([x, y], k) => { w.set(bigToWords(x), k * 16); w.set(bigToWords(y), k * 16 + 8); });
    return w;
}

// ---- patterns ----
// Letter-case classes for case-insensitive matching (digit index -> class id).
export const FOLD = Array.from(B58, (ch) => {
    const lo = ch.toLowerCase(), up = ch.toUpperCase();
    if (lo !== up && B58.includes(lo) && B58.includes(up)) return B58.indexOf(lo);
    return B58.indexOf(ch);
});

function variants(s, icase) {
    let out = [""];
    for (const ch of s) {
        let opts;
        if (icase && /[a-z]/i.test(ch)) opts = [ch.toLowerCase(), ch.toUpperCase()].filter((c, i, a) => B58.includes(c) && a.indexOf(c) === i);
        else opts = B58.includes(ch) ? [ch] : [];
        const next = [];
        for (const o of out) for (const c of opts) next.push(o + c);
        out = next;
        if (out.length > 65536) throw new Error("too many case variants");
    }
    return out;
}
const numOf = (s) => { let v = 0n; for (const c of s) v = v * 58n + BigInt(B58.indexOf(c)); return v; };

// Every Tron address integer is in [0x41 * 2^192, 0x42 * 2^192): 34 characters, first one T.
const LO0 = 0x41n << 192n, HI0 = 0x42n << 192n;
export function normPrefix(p, icase) {
    if (!p) return p;
    if (p[0] === "T" || (icase && p[0] === "t")) return "T" + p.slice(1);
    return "T" + p;
}

// patterns: [{prefix, suffix}] (prefix without the leading T is fine), returns what the GPU and the UI need
export function buildPatterns(input, icase, { forceSkip = false } = {}) {
    const bitmap = new Uint32Array(8);
    const list = [];
    let ranges = [], prob = 0, needCk = false;
    const targets = [], sufLens = [];
    for (const { prefix: p0 = "", suffix = "" } of input) {
        const prefix = normPrefix(p0, icase);
        if (!prefix && !suffix) throw new Error("empty pattern");
        if (prefix.length + suffix.length > 34) throw new Error("pattern too long");
        for (const s of [prefix, suffix]) {
            if (s && variants(s, icase).length === 0)
                throw new Error(`"${s}" has characters outside base58 (0, O, I, l are not used)`);
        }
        let ppre = 1;
        if (prefix) {
            let tot = 0n;
            const rr = [];
            for (const v of variants(prefix, icase)) {
                const step = 58n ** BigInt(34 - v.length);
                let lo = numOf(v) * step, hi = lo + step;
                if (lo < LO0) lo = LO0;
                if (hi > HI0) hi = HI0;
                if (lo >= hi) continue;
                tot += hi - lo; rr.push([lo, hi]);
            }
            if (!tot) throw new Error(`no Tron address can start with ${prefix}`);
            ppre = Number(tot * 1000000000000n / (HI0 - LO0)) / 1e12;
            if (!suffix) ranges = ranges.concat(rr);
        }
        let psuf = 1;
        if (suffix) {
            needCk = true;
            const vs = variants(suffix, icase), K = suffix.length;
            psuf = vs.length / 58 ** K;
            const kb = Math.min(K, 8), m = (1 << kb) - 1;
            sufLens.push(K);
            for (const v of vs) {
                if (K <= 10) targets.push(numOf(v));
                const low = Number(numOf(v) & 0xffn);
                for (let b = 0; b < 256; b++) if ((b & m) === (low & m)) bitmap[b >> 5] |= 1 << (b & 31);
            }
        }
        prob += ppre * psuf;
        list.push({ prefix, suffix });
    }
    // gate ranges on the top 64 bits (N >> 136), merged; shrunk to 64 by joining neighbours (superset)
    const r64 = ranges.map(([lo, hi]) => [lo >> 136n, (hi - 1n) >> 136n]).sort((a, b) => (a[0] < b[0] ? -1 : 1));
    const merged = [];
    for (const r of r64) {
        if (merged.length && r[0] <= merged[merged.length - 1][1] + 1n) {
            if (r[1] > merged[merged.length - 1][1]) merged[merged.length - 1][1] = r[1];
        } else merged.push([r[0], r[1]]);
    }
    while (merged.length > 64) {
        let best = 0, gap = null;
        for (let i = 0; i + 1 < merged.length; i++) {
            const g = merged[i + 1][0] - merged[i][1];
            if (gap === null || g < gap) { gap = g; best = i; }
        }
        merged[best][1] = merged[best + 1][1];
        merged.splice(best + 1, 1);
    }
    const skip = planSkip(targets, sufLens, forceSkip);
    if (skip) needCk = false;
    return { list, icase, bitmap, ranges: merged, prob, needCk, skip };
}

// Checksum skip (see the kernel): every suffix has the same length K in 6..10 and few enough values.
export function planSkip(targets, sufLens, force = false) {
    if (!targets.length || sufLens.some((k) => k !== sufLens[0])) return null;
    const K = sufLens[0];
    if (K < 6 || K > 10 || targets.length > 128) return null;
    if (!force && targets.length * 2 ** 32 / 58 ** K > 0.5) return null;
    return { K, m: 58n ** BigInt(K), targets };
}
// (A * 2^32) mod m for the 25-byte address bytes (A = first 21 bytes)
export function skipResidue(addr, m) {
    let a = 0n;
    for (let i = 0; i < 21; i++) a = (a << 8n) | BigInt(addr[i]);
    return (a << 32n) % m;
}
// GPU buffer: m * 1,2,4,8,16 | ntarget | 0x41 term | targets | per-byte residues of the 20 hash bytes
export function skipWords(skip, tabOff, maxTargets) {
    const w = new Uint32Array(tabOff + 20 * 256 * 2);
    if (!skip) return w;
    const put = (i, v) => { w[i] = Number(v & 0xffffffffn); w[i + 1] = Number(v >> 32n); };
    const { m, targets } = skip;
    [1n, 2n, 4n, 8n, 16n].forEach((s, k) => put(2 * k, s * m));
    w[10] = targets.length;
    put(12, (0x41n << 192n) % m);
    targets.slice(0, maxTargets).forEach((t, v) => put(16 + 2 * v, t % m));
    for (let i = 0; i < 20; i++) {
        const wgt = (1n << BigInt(8 * (19 - i) + 32)) % m;
        for (let b = 0; b < 256; b++) put(tabOff + (i * 256 + b) * 2, BigInt(b) * wgt % m);
    }
    return w;
}

export function matches(addr, pats) {
    const f = pats.icase ? (s) => s.toLowerCase() : (s) => s;
    const a = f(addr);
    return pats.list.some(({ prefix = "", suffix = "" }) => a.startsWith(f(prefix)) && a.endsWith(f(suffix)));
}

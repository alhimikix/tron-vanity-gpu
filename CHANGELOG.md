# Changelog

## Browser version - 2026-11-10

- WebGPU page (`web/`, https://alhimikix.github.io/tron-vanity-gpu/): prefix, suffix or both on the GPU of any
  device with WebGPU; ~840M keys/sec on an RTX 5080 in Chrome (prefixes, suffixes of 6+ characters), ~360M for
  shorter suffixes.
- secp256k1 field on 20 x 13-bit limbs in WGSL (`tools/gen_wgsl_secp.py`), Keccak-256 and double SHA-256 on the
  GPU, checksum skip with a per-thread queue.
- Every match re-checked with a BigInt reference before it is shown; in-page selftest; CSV/JSON export.

## v1.1.0 - 2026-11-10

- Suffixes of 6+ characters skip the double SHA-256 for almost every key: the needed checksum follows from the
  address, and it is out of range for 89% (K = 6) to 99.997% (K = 8) of keys. 6 characters: 1.12 -> ~1.45 billion
  keys/sec, 8 characters: 1.12 -> ~1.75 billion on an RTX 5080.
- Keys that still need the checksum are queued per warp and hashed 32 at a time.
- Host check of matches: Jacobian k*G over a 2^i table and several threads (short patterns no longer slow the
  search down).
- Selftest also checks the checksum-skip residues and the fast host k*G.

## v1.0.0 - 2026-11-10

First release.

- Tron vanity addresses on NVIDIA GPUs: prefix, suffix or both, several patterns at once, case-insensitive.
- ~1.1 billion keys/sec on an RTX 5080 (suffix search), ~1.7 billion for prefix-only search.
- Random starts from ChaCha12 keyed by the OS RNG; neighbouring keys by batched affine additions.
- secp256k1 field arithmetic in PTX (Karatsuba multiply), Keccak-256 and double SHA-256 on the GPU.
- Every match re-derived and re-checked on the CPU; `--selftest`; `tools/tron_ref.py` for independent checks.
- Odds and expected time per match; CSV output `address,private_key`.

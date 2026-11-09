# Changelog

## v1.0.0 - 2026-11-10

First release.

- Tron vanity addresses on NVIDIA GPUs: prefix, suffix or both, several patterns at once, case-insensitive.
- ~1.1 billion keys/sec on an RTX 5080 (suffix search), ~1.7 billion for prefix-only search.
- Random starts from ChaCha12 keyed by the OS RNG; neighbouring keys by batched affine additions.
- secp256k1 field arithmetic in PTX (Karatsuba multiply), Keccak-256 and double SHA-256 on the GPU.
- Every match re-derived and re-checked on the CPU; `--selftest`; `tools/tron_ref.py` for independent checks.
- Odds and expected time per match; CSV output `address,private_key`.

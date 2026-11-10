# Emits WGSL for the secp256k1 field (p = 2^256 - 2^32 - 977) on 20 x 13-bit limbs (u32), used by the
# browser (WebGPU) grinder. value = sum r[i] * 2^(13 i); after fe_norm limbs are < 2^13 (+ a small
# carry on limbs 0..3) and limb 19 keeps 9 bits, so values are < 2^256 + small.
# Multiplication is product scanning (a column holds at most 20 products < 2^26.4, no u32 overflow).
# Column weights 2^260 and up fold back with 2^260 = 2^36 + 15632 (mod p); the 2^256 overflow of
# the top limb folds with 2^256 = 2^32 + 977.
#
# Usage: python tools/gen_wgsl_secp.py > web/js/fe.wgsl.js
N, B, M = 20, 13, (1 << 13) - 1
P = (1 << 256) - (1 << 32) - 977


def norm_lines(r, indent="    "):
    L = []
    for p in range(2):
        L.append("c = 0u;")
        for i in range(N - 1):
            L.append(f"{r}[{i}] += c; c = {r}[{i}] >> 13u; {r}[{i}] &= 0x1fffu;")
        L.append(f"{r}[19] += c; c = {r}[19] >> 9u; {r}[19] &= 0x1ffu;")
        L.append(f"{r}[0] += 977u * c; {r}[2] += c << 6u;")
    return [indent + l for l in L]


def mul(square=False):
    name = "fe_sq" if square else "fe_mul"
    sig = "(a: Fe) -> Fe" if square else "(a: Fe, b: Fe) -> Fe"
    L = [f"fn {name}{sig} {{", "    var r: array<u32, 22>;", "    var c: u32 = 0u;", "    var t: u32;", "    var h: u32;"]
    for k in range(2 * N - 1):
        terms = []
        for i in range(max(0, k - N + 1), min(k, N - 1) + 1):
            j = k - i
            if square:
                if i > j: continue
                terms.append(f"a[{i}] * a[{j}]" if i == j else f"2u * (a[{i}] * a[{j}])")
            else:
                terms.append(f"a[{i}] * b[{j}]")
        L.append(f"    t = c + " + " + ".join(terms) + ";")
        if k < N:
            L.append(f"    r[{k}] += t & 0x1fffu; c = t >> 13u;")
        else:
            L.append(f"    h = t & 0x1fffu; c = t >> 13u; r[{k - N}] += 15632u * h; r[{k - N + 2}] += 1024u * h;")
    # remaining carry: weight 2^(13*39) = 2^(13*19) * 2^260
    L.append("    r[19] += 15632u * c; r[21] += 1024u * c;")
    # carry over 22 limbs, then fold limbs 20, 21 (weights 2^260, 2^273)
    L.append("    c = 0u;")
    for i in range(21):
        L.append(f"    r[{i}] += c; c = r[{i}] >> 13u; r[{i}] &= 0x1fffu;")
    L.append("    r[21] += c;")
    L.append("    r[0] += 15632u * r[20]; r[2] += 1024u * r[20];")
    L.append("    r[1] += 15632u * r[21]; r[3] += 1024u * r[21];")
    L.append("    var o: Fe;")
    for i in range(N):
        L.append(f"    o[{i}] = r[{i}];")
    L += [l.replace("{r}", "o") for l in norm_lines("o")]
    L += ["    return o;", "}"]
    return L


def sub_const():
    # 8p with every limb >= 2^13 + 256 (limb 19 well above 2^9), so a + K - b never underflows
    v = 8 * P
    d = [(v >> (B * i)) & M for i in range(N)]
    d[19] = v >> (B * 19)
    for i in range(N - 1):
        d[i] += 2 << B
        d[i + 1] -= 2
    assert sum(x << (B * i) for i, x in enumerate(d)) == v
    assert all(x >= (1 << B) + 1100 for x in d[:19]) and d[19] >= 2000, d
    return d


def pack():
    # exact 256-bit value after fe_carry -> 8 little-endian u32
    L = ["fn fe_pack(a0: Fe) -> array<u32, 8> {", "    let a = fe_carry(a0);", "    var w: array<u32, 8>;"]
    for k in range(8):
        parts = []
        for i in range(N):
            lo, hi = B * i, B * i + B
            if hi <= 32 * k or lo >= 32 * k + 32: continue
            sh = lo - 32 * k
            parts.append(f"(a[{i}] << {sh}u)" if sh >= 0 else f"(a[{i}] >> {-sh}u)")
        L.append(f"    w[{k}] = " + " | ".join(parts) + ";")
    L += ["    return w;", "}"]
    return L


def unpack():
    L = ["fn fe_unpack(w: array<u32, 8>) -> Fe {", "    var r: Fe;"]
    for i in range(N):
        lo = B * i
        k, sh = lo // 32, lo % 32
        width = 9 if i == 19 else B
        expr = f"(w[{k}] >> {sh}u)"
        if sh + width > 32 and k + 1 < 8:
            expr = f"((w[{k}] >> {sh}u) | (w[{k + 1}] << {32 - sh}u))"
        mask = (1 << width) - 1
        L.append(f"    r[{i}] = {expr} & {hex(mask)}u;")
    L += ["    return r;", "}"]
    return L


def main():
    NL = "\n"
    K = sub_const()
    out = ["alias Fe = array<u32, 20>;",
           "const FE_SUBK = Fe(" + ", ".join(f"{x}u" for x in K) + ");",
           "fn fe_norm(a: Fe) -> Fe {" + NL + "    var r = a;" + NL + "    var c: u32 = 0u;" + NL
           + NL.join(norm_lines("r")) + NL + "    return r;" + NL + "}",
           "fn fe_add(a: Fe, b: Fe) -> Fe {" + NL + "    var r: Fe;" + NL
           + NL.join(f"    r[{i}] = a[{i}] + b[{i}];" for i in range(N)) + NL + "    return fe_norm(r);" + NL + "}",
           "fn fe_sub(a: Fe, b: Fe) -> Fe {" + NL + "    var r: Fe;" + NL
           + NL.join(f"    r[{i}] = a[{i}] + FE_SUBK[{i}] - b[{i}];" for i in range(N)) + NL + "    return fe_norm(r);" + NL + "}",
           # exact carry: limbs 0..18 < 2^13, limb 19 holds the rest (value < 2^256 + small)
           "fn fe_carry(a: Fe) -> Fe {" + NL + "    var r = fe_norm(a);" + NL + "    var c: u32 = 0u;" + NL
           + NL.join(f"    r[{i}] += c; c = r[{i}] >> 13u; r[{i}] &= 0x1fffu;" for i in range(N - 1))
           + NL + "    r[19] += c;" + NL + "    return r;" + NL + "}",
           NL.join(mul(False)), NL.join(mul(True)), NL.join(pack()), NL.join(unpack())]
    print("// Generated by tools/gen_wgsl_secp.py - do not edit.")
    print("export const FE_WGSL = `")
    print((NL + NL).join(out))
    print("`;")


main()


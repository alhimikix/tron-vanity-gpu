# Emits PTX-asm CUDA functions for the secp256k1 field, p = 2^256 - 2^32 - 977, on 8 x 32-bit limbs.
# Values are kept lazily in [0, 2^256); 2^256 = 2^32 + 977 (mod p).
# fe_mul: one level of Karatsuba on 128-bit halves (48 products) with row-wise carry chains.
# Usage: python tools/gen_secp.py > fe.h, then paste between the GENERATED markers in vanity.cu.
def mul():
    L=[]; a=lambda i:f"%{8+i}"; b=lambda j:f"%{16+j}"; t=lambda k:f"t{k}"
    L.append(".reg .u32 t<17>;")
    for k in range(17): L.append(f"mov.u32 t{k}, 0;")
    for i in range(8):
        for j in range(8):
            op = "mad.lo.cc.u32" if j == 0 else "madc.lo.cc.u32"
            L.append(f"{op} {t(i+j)}, {a(i)}, {b(j)}, {t(i+j)};")
        L.append(f"addc.u32 {t(i+8)}, {t(i+8)}, 0;")
        for j in range(8):
            op = "mad.hi.cc.u32" if j == 0 else ("madc.hi.cc.u32" if j < 7 else "madc.hi.u32")
            L.append(f"{op} {t(i+j+1)}, {a(i)}, {b(j)}, {t(i+j+1)};")
    L += red("t")
    return L
def sqr():
    L=[]; a=lambda i:f"%{8+i}"; t=lambda k:f"t{k}"
    L.append(".reg .u32 t<17>;")
    for k in range(17): L.append(f"mov.u32 t{k}, 0;")
    # off-diagonal a_i*a_j, i<j
    for i in range(7):
        js=list(range(i+1,8))
        for n,j in enumerate(js):
            op = "mad.lo.cc.u32" if n == 0 else "madc.lo.cc.u32"
            L.append(f"{op} {t(i+j)}, {a(i)}, {a(j)}, {t(i+j)};")
        L.append(f"addc.u32 {t(i+8)}, {t(i+8)}, 0;")
        for n,j in enumerate(js):
            last = n == len(js)-1
            op = "mad.hi.cc.u32" if n == 0 else "madc.hi.cc.u32"
            if last: op = op.replace(".cc","") if n>0 else "mad.hi.cc.u32"
            L.append(f"{op} {t(i+j+1)}, {a(i)}, {a(j)}, {t(i+j+1)};")
            if last and n == 0: L.append(f"addc.u32 {t(i+j+2)}, {t(i+j+2)}, 0;")
    # double
    L.append("add.cc.u32 t1, t1, t1;")
    for k in range(2,16): L.append(f"addc.cc.u32 t{k}, t{k}, t{k};")
    L.append("addc.u32 t16, 0, 0;")
    # diagonals
    for i in range(8):
        op_lo = "mad.lo.cc.u32" if i == 0 else "madc.lo.cc.u32"
        L.append(f"{op_lo} {t(2*i)}, {a(i)}, {a(i)}, {t(2*i)};")
        op_hi = "madc.hi.cc.u32" if i < 7 else "madc.hi.u32"
        L.append(f"{op_hi} {t(2*i+1)}, {a(i)}, {a(i)}, {t(2*i+1)};")
    L += red("t")
    return L
def emit(name, body, nin, clob=False):
    outs = ", ".join(('"=&r"' if clob else '"=r"') + f'(r[{k}])' for k in range(8))
    ins = ", ".join([f'"r"(a[{k}])' for k in range(8)] + ([f'"r"(b[{k}])' for k in range(8)] if nin==2 else [f'"r"(a[{k}])' for k in range(8)]))
    sig = "u32 r[8], const u32 a[8], const u32 b[8]" if nin==2 else "u32 r[8], const u32 a[8]"
    BS = chr(92); NL = chr(10)
    asm = NL.join('        "' + l + BS + "n" + BS + 't"' for l in ["{"] + body + ["}"])
    return f"__device__ __forceinline__ void {name}({sig}) {{\n    asm volatile(\n{asm}\n        : {outs}\n        : {ins});\n}}\n"


def prod(L, dst, A, B):
    # dst (2n regs, zero-initialised by caller) = A * B, n-limb row-wise with carry chains
    n = len(A)
    for i in range(n):
        for j in range(n):
            op = "mad.lo.cc.u32" if j == 0 else "madc.lo.cc.u32"
            L.append(f"{op} {dst[i+j]}, {A[i]}, {B[j]}, {dst[i+j]};")
        L.append(f"addc.u32 {dst[i+n]}, {dst[i+n]}, 0;")
        for j in range(n):
            op = "mad.hi.cc.u32" if j == 0 else ("madc.hi.cc.u32" if j < n-1 else "madc.hi.u32")
            L.append(f"{op} {dst[i+j+1]}, {A[i]}, {B[j]}, {dst[i+j+1]};")

_ctr = [0]
def kara(L, decl, dst, A, B, depth):
    n = len(A)
    if depth == 0 or n <= 2:
        prod(L, dst, A, B); return
    h = n // 2
    kara(L, decl, dst[0:n], A[:h], B[:h], depth - 1)
    kara(L, decl, dst[n:2*n], A[h:], B[h:], depth - 1)
    _ctr[0] += 1; q = f"q{_ctr[0]}_"
    sa = [f"{q}a{i}" for i in range(h+1)]; sb = [f"{q}b{i}" for i in range(h+1)]
    m = [f"{q}m{i}" for i in range(n+1)]; k = f"{q}k"; mk = f"{q}mk"
    decl += sa + sb + m + [k, mk]
    for r in m: L.append(f"mov.u32 {r}, 0;")
    for X, S in ((A, sa), (B, sb)):
        L.append(f"add.cc.u32 {S[0]}, {X[0]}, {X[h]};")
        for i in range(1, h): L.append(f"addc.cc.u32 {S[i]}, {X[i]}, {X[i+h]};")
        L.append(f"addc.u32 {S[h]}, 0, 0;")
    kara(L, decl, m[0:n], sa[:h], sb[:h], depth - 1)
    for c, o in ((sa[h], sb), (sb[h], sa)):
        L.append(f"neg.s32 {mk}, {c};")
        for i in range(h):
            L.append(f"and.b32 {k}, {o[i]}, {mk};")
            L.append(f"{'add.cc.u32' if i == 0 else 'addc.cc.u32'} {m[h+i]}, {m[h+i]}, {k};")
        L.append(f"addc.u32 {m[n]}, {m[n]}, 0;")
    L.append(f"and.b32 {k}, {sa[h]}, {sb[h]};")
    L.append(f"add.u32 {m[n]}, {m[n]}, {k};")
    for part in (dst[0:n], dst[n:2*n]):
        for i in range(n):
            L.append(f"{'sub.cc.u32' if i == 0 else 'subc.cc.u32'} {m[i]}, {m[i]}, {part[i]};")
        L.append(f"subc.u32 {m[n]}, {m[n]}, 0;")
    tail = dst[h:]
    for i in range(len(tail)):
        if i <= n:
            L.append(f"{'add.cc.u32' if i == 0 else 'addc.cc.u32'} {tail[i]}, {tail[i]}, {m[i]};")
        else:
            L.append(f"addc.cc.u32 {tail[i]}, {tail[i]}, 0;")
    L.append(f"addc.u32 {k}, 0, 0;")   # final carry is always 0 (product fits)

def mul_kd(depth):
    decl = [f"t{i}" for i in range(16)]
    L = []
    a = [f"%{8+i}" for i in range(8)]; b = [f"%{16+i}" for i in range(8)]
    t = [f"t{i}" for i in range(16)]
    for r in t: L.append(f"mov.u32 {r}, 0;")
    kara(L, decl, t, a, b, depth)
    L += red("t")
    return [".reg .u32 " + ", ".join(decl) + ";"] + L

def red(t):
    # r = t_lo + t_hi * (2^32 + 977), folded until it fits 256 bits
    L = [".reg .u32 rc, rch, rf, rg;"]
    r = lambda k: f"%{k}"
    T = lambda k: f"{t}{k}"
    # r = t_lo + 977 * t_hi
    for k in range(8):
        L.append(f"{'mad.lo.cc.u32' if k == 0 else 'madc.lo.cc.u32'} {r(k)}, {T(8+k)}, 977, {T(k)};")
    L.append("addc.u32 rc, 0, 0;")
    for k in range(8):
        dst = r(k + 1) if k < 7 else "rc"
        op = "mad.hi.cc.u32" if k == 0 else ("madc.hi.cc.u32" if k < 7 else "madc.hi.u32")
        L.append(f"{op} {dst}, {T(8+k)}, 977, {dst};")
    # += t_hi << 32
    for k in range(7):
        L.append(f"{'add.cc.u32' if k == 0 else 'addc.cc.u32'} {r(k+1)}, {r(k+1)}, {T(8+k)};")
    L.append(f"addc.cc.u32 rc, rc, {T(15)};")
    L.append("addc.u32 rch, 0, 0;")
    # value = r + (rc + rch*2^32) * 2^256 ; fold C*(2^32+977) = rc*977 + (rc + rch*977)*2^32 + rch*2^64
    L.append(f"mad.lo.cc.u32 {r(0)}, rc, 977, {r(0)};")
    L.append(f"madc.hi.cc.u32 {r(1)}, rc, 977, {r(1)};")
    for k in range(2, 8): L.append(f"addc.cc.u32 {r(k)}, {r(k)}, 0;")
    L.append("addc.u32 rf, 0, 0;")
    L.append(f"add.cc.u32 {r(1)}, {r(1)}, rc;")
    L.append(f"addc.cc.u32 {r(2)}, {r(2)}, rch;")
    for k in range(3, 8): L.append(f"addc.cc.u32 {r(k)}, {r(k)}, 0;")
    L.append("addc.u32 rf, rf, 0;")
    L.append(f"mad.lo.cc.u32 {r(1)}, rch, 977, {r(1)};")
    for k in range(2, 8): L.append(f"addc.cc.u32 {r(k)}, {r(k)}, 0;")
    L.append("addc.u32 rf, rf, 0;")
    # value = r + rf * 2^256, rf <= 3
    L.append(f"mad.lo.cc.u32 {r(0)}, rf, 977, {r(0)};")
    L.append(f"addc.cc.u32 {r(1)}, {r(1)}, rf;")
    for k in range(2, 8): L.append(f"addc.cc.u32 {r(k)}, {r(k)}, 0;")
    L.append("addc.u32 rg, 0, 0;")
    # rare second wrap: r is tiny now, so this cannot carry out again
    L.append(f"mad.lo.cc.u32 {r(0)}, rg, 977, {r(0)};")
    L.append(f"addc.cc.u32 {r(1)}, {r(1)}, rg;")
    for k in range(2, 7): L.append(f"addc.cc.u32 {r(k)}, {r(k)}, 0;")
    L.append(f"addc.u32 {r(7)}, {r(7)}, 0;")
    return L

def add():
    L = [".reg .u32 c;"]; a = lambda i: f"%{8+i}"; b = lambda j: f"%{16+j}"; r = lambda k: f"%{k}"
    for k in range(8):
        L.append(f"{'add.cc.u32' if k == 0 else 'addc.cc.u32'} {r(k)}, {a(k)}, {b(k)};")
    for _ in range(2):   # carry out = +2^256 = +(2^32 + 977); the second pass can only follow a tiny wrap
        L.append("addc.u32 c, 0, 0;")
        L.append(f"mad.lo.cc.u32 {r(0)}, c, 977, {r(0)};")
        L.append(f"addc.cc.u32 {r(1)}, {r(1)}, c;")
        for k in range(2, 8): L.append(f"addc.cc.u32 {r(k)}, {r(k)}, 0;")
    return L

def sub():
    L = [".reg .u32 c, m, n;"]; a = lambda i: f"%{8+i}"; b = lambda j: f"%{16+j}"; r = lambda k: f"%{k}"
    for k in range(8):
        L.append(f"{'sub.cc.u32' if k == 0 else 'subc.cc.u32'} {r(k)}, {a(k)}, {b(k)};")
    for _ in range(2):   # borrow = we added 2^256, i.e. (2^32 + 977) too much
        L.append("subc.u32 c, 0, 0;")
        L.append("and.b32 m, c, 977;")
        L.append("and.b32 n, c, 1;")
        L.append(f"sub.cc.u32 {r(0)}, {r(0)}, m;")
        L.append(f"subc.cc.u32 {r(1)}, {r(1)}, n;")
        for k in range(2, 8): L.append(f"subc.cc.u32 {r(k)}, {r(k)}, 0;")
    return L

print("// Generated by tools/gen_secp.py - do not edit." + chr(10) + emit("fe_mul", mul_kd(1), 2) + emit("fe_sq", sqr(), 1) + emit("fe_add", add(), 2, True) + emit("fe_sub", sub(), 2, True))

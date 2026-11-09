# Reference Tron address derivation (pure Python secp256k1 + Keccak-256 from pycryptodome).
# Usage: python tools/tron_ref.py <private_key_hex>          -> prints the address
#        python tools/tron_ref.py --verify results.csv [--prefix P] [--suffix S] [--icase]
import hashlib, sys
from Crypto.Hash import keccak

P = 2**256 - 2**32 - 977
N = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
G = (0x79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798,
     0x483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8)
B58 = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"


def add(p, q):
    if p is None: return q
    if q is None: return p
    if p[0] == q[0] and (p[1] + q[1]) % P == 0: return None
    if p == q: l = 3 * p[0] * p[0] * pow(2 * p[1], P - 2, P) % P
    else: l = (q[1] - p[1]) * pow(q[0] - p[0], P - 2, P) % P
    x = (l * l - p[0] - q[0]) % P
    return (x, (l * (p[0] - x) - p[1]) % P)


def mul(k):
    r, q = None, G
    while k:
        if k & 1: r = add(r, q)
        q = add(q, q); k >>= 1
    return r


def b58(b):
    n = int.from_bytes(b, "big"); s = ""
    while n: n, m = divmod(n, 58); s = B58[m] + s
    return "1" * (len(b) - len(b.lstrip(b"\0"))) + s


def address(priv: int) -> str:
    x, y = mul(priv)
    h = keccak.new(digest_bits=256, data=x.to_bytes(32, "big") + y.to_bytes(32, "big")).digest()
    a = b"\x41" + h[12:]
    return b58(a + hashlib.sha256(hashlib.sha256(a).digest()).digest()[:4])


def eth_address(priv: int) -> str:
    x, y = mul(priv)
    return keccak.new(digest_bits=256, data=x.to_bytes(32, "big") + y.to_bytes(32, "big")).hexdigest()[24:]


if __name__ == "__main__":
    args = sys.argv[1:]
    if args and args[0] == "--verify":
        path = args[1]; pre = []; suf = []; icase = "--icase" in args
        for i, a in enumerate(args):
            if a == "--prefix": pre.append(args[i + 1])
            if a == "--suffix": suf.append(args[i + 1])
        f = (lambda s: s.lower()) if icase else (lambda s: s)
        ok = bad = 0
        for line in open(path):
            line = line.strip()
            if not line or line.startswith("address"): continue
            addr, key = line.split(",")[:2]
            good = address(int(key, 16)) == addr
            if pre or suf:
                pp = [p if p.startswith("T") else "T" + p for p in pre]
                good = good and (any(f(addr).startswith(f(p)) for p in pp) or any(f(addr).endswith(f(s)) for s in suf))
            ok += good; bad += not good
            if not good: print("BAD", addr)
        print(f"verified ok={ok} bad={bad}")
    else:
        k = int(args[0], 16) if args else 1
        print(address(k), eth_address(k))

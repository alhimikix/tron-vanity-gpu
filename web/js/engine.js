// WebGPU Tron vanity grinder engine (browser and Node with a WebGPU implementation).
import { shader, M, STEP, ROUNDS, REC, PSTRIDE, SKIP_TAB, SKIP_TARGETS } from "./kernels.js";
import * as R from "./ref.js";

const MATCH_CAP = 4096;
const VERIFY_CAP = 16;   // matches checked and shown per batch

function limbs(x) {   // 20 x 13-bit limbs
    const out = new Uint32Array(20);
    for (let i = 0; i < 20; i++) { out[i] = Number(x & 0x1fffn); x >>= 13n; }
    return out;
}

export class Grinder {
    // Each GPU thread owns one random start ("chunk") and walks ROUNDS rounds of STEP keys from it.
    // threads: threads per dispatch; the batch grows (more threads, then more rounds per dispatch) while
    // a batch takes < 100 ms, since browsers add per-batch latency but slow GPUs must keep batches short.
    static async create({ gpu = globalThis.navigator?.gpu, threads = 4096, maxThreads = 65536, rounds = 1, dump = false } = {}) {
        if (!gpu) throw new Error("WebGPU is not available in this browser");
        const adapter = await gpu.requestAdapter({ powerPreference: "high-performance" });
        if (!adapter) throw new Error("No WebGPU adapter found");
        const device = await adapter.requestDevice({
            requiredLimits: { maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize, maxBufferSize: adapter.limits.maxBufferSize },
        });
        const g = new Grinder();
        g.adapter = adapter; g.device = device;
        // any validation / out-of-memory error must stop the run, never be skipped silently
        device.addEventListener("uncapturederror", (ev) => { g.gpuError = ev.error?.message || String(ev.error); });
        device.lost.then((info) => { g.gpuError = g.gpuError || `GPU device lost: ${info.message || info.reason}`; });
        g.info = adapter.info || {};
        g.T = threads; g.capT = Math.max(threads, maxThreads); g.R = rounds;
        g.dumpMode = dump;
        await g._init();
        return g;
    }

    async _init() {
        const d = this.device;
        const t0 = performance.now();
        const mod = d.createShaderModule({ code: shader() });
        await this._checkModule(mod);
        const S = GPUShaderStage.COMPUTE;
        const ro = { type: "read-only-storage" }, rw = { type: "storage" };
        this.bgl = d.createBindGroupLayout({ entries: [
            { binding: 0, visibility: S, buffer: { type: "uniform" } },
            { binding: 1, visibility: S, buffer: ro }, { binding: 2, visibility: S, buffer: rw },
            { binding: 3, visibility: S, buffer: rw }, { binding: 4, visibility: S, buffer: ro },
            { binding: 5, visibility: S, buffer: ro }, { binding: 6, visibility: S, buffer: rw },
            { binding: 7, visibility: S, buffer: rw }, { binding: 8, visibility: S, buffer: ro }] });
        const layout = d.createPipelineLayout({ bindGroupLayouts: [this.bgl] });
        const [startPipe, walkPipe] = await Promise.all([
            d.createComputePipelineAsync({ layout, compute: { module: mod, entryPoint: "start_main" } }),
            d.createComputePipelineAsync({ layout, compute: { module: mod, entryPoint: "walk_main" } })]);
        this.startPipe = startPipe; this.walkPipe = walkPipe;
        this.compileMs = performance.now() - t0;

        // tables: j*G (j = 1..M), STEP*G, then d * 16^w * G; as 13-bit limbs
        const pts = [...R.walkTable(M, STEP), ...R.windowTable()];
        const tw = new Uint32Array(pts.length * 40);
        pts.forEach(([x, y], k) => { tw.set(limbs(x), k * 40); tw.set(limbs(y), k * 40 + 20); });
        this.tab = this._buf(tw.byteLength, GPUBufferUsage.STORAGE, tw);

        const ST = GPUBufferUsage.STORAGE;
        this.CX = this._buf(this.capT * 32, ST); this.CY = this._buf(this.capT * 32, ST);
        this.params = this._buf(112, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
        this.outBytes = (4 + MATCH_CAP * REC) * 4;
        this.out = this._buf(this.outBytes, ST | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST);
        this.dumpBytes = this.dumpMode ? this.T * this.R * STEP * 36 : 16;
        this.dump = this._buf(this.dumpBytes, ST | GPUBufferUsage.COPY_SRC);
        this.key = new Uint32Array(8);
        crypto.getRandomValues(this.key);
        this.next = 0n;         // next unused chunk counter
        this.round = ROUNDS;    // forces fresh starts on the first batch
        this.initMs = performance.now() - t0;
    }

    async _checkModule(m) {
        if (!m.getCompilationInfo) return;
        const info = await m.getCompilationInfo();
        const errs = info.messages.filter((x) => x.type === "error");
        if (errs.length) throw new Error("shader: " + errs.map((e) => `${e.lineNum}:${e.linePos} ${e.message}`).join("\n"));
    }

    _buf(size, usage, data) {
        const b = this.device.createBuffer({ size: Math.max(16, Math.ceil(size / 16) * 16), usage, mappedAtCreation: !!data });
        if (data) { new Uint32Array(b.getMappedRange()).set(data); b.unmap(); }
        return b;
    }

    setPatterns(list, icase, opts) {
        const pats = R.buildPatterns(list, icase, opts);
        const pw = new Uint32Array(Math.max(1, pats.list.length) * PSTRIDE);
        const enc = (c) => { let v = R.B58.indexOf(c); if (v < 0) v = R.B58.indexOf(c.toLowerCase()); if (v < 0) v = R.B58.indexOf(c.toUpperCase()); return icase ? R.FOLD[v] : v; };
        pats.list.forEach(({ prefix, suffix }, i) => {
            pw[i * PSTRIDE] = prefix.length; pw[i * PSTRIDE + 1] = suffix.length;
            [...prefix].forEach((c, q) => { pw[i * PSTRIDE + 2 + q] = enc(c); });
            [...suffix].forEach((c, q) => { pw[i * PSTRIDE + 36 + q] = enc(c); });
        });
        const rw = new Uint32Array(Math.max(1, pats.ranges.length) * 4);
        pats.ranges.forEach(([lo, hi], i) => {
            rw[i * 4] = Number(lo >> 32n); rw[i * 4 + 1] = Number(lo & 0xffffffffn);
            rw[i * 4 + 2] = Number(hi >> 32n); rw[i * 4 + 3] = Number(hi & 0xffffffffn);
        });
        this.pats = pats;
        this.patBuf?.destroy(); this.rangeBuf?.destroy(); this.skipBuf?.destroy();
        this.skipBuf = this._buf((SKIP_TAB + 20 * 256 * 2) * 4, GPUBufferUsage.STORAGE, R.skipWords(pats.skip, SKIP_TAB, SKIP_TARGETS));
        this.patBuf = this._buf(pw.byteLength, GPUBufferUsage.STORAGE, pw);
        this.rangeBuf = this._buf(rw.byteLength, GPUBufferUsage.STORAGE, rw);
        this.bg = this.device.createBindGroup({ layout: this.bgl, entries: [
            this.params, this.tab, this.CX, this.CY, this.patBuf, this.rangeBuf, this.out, this.dump, this.skipBuf,
        ].map((buffer, binding) => ({ binding, resource: { buffer } })) });
        return pats;
    }

    _params(base, round0) {
        const u = new Uint32Array(28);
        u.set(this.key, 0);
        u.set(this.pats.bitmap, 8);
        u[16] = Number(base & 0xffffffffn); u[17] = Number(base >> 32n); u[18] = this.T; u[19] = this.R;
        u[20] = round0; u[21] = this.pats.list.length; u[22] = this.pats.icase ? 1 : 0; u[23] = this.pats.ranges.length;
        u[24] = MATCH_CAP; u[25] = this.pats.needCk ? 1 : 0; u[26] = this.dumpMode ? 1 : 0; u[27] = this.pats.skip ? 1 : 0;
        return u;
    }

    // Enqueue one batch; resolves with { base, round0, keys, count, records } once results are readable.
    _batch() {
        const d = this.device, T = this.T, Rn = this.R;
        const enc = d.createCommandEncoder();
        if (this.round + Rn > ROUNDS) {   // new random starts
            this.base = this.next; this.next += BigInt(T); this.round = 0;
            d.queue.writeBuffer(this.params, 0, this._params(this.base, 0));
            const p = enc.beginComputePass();
            p.setPipeline(this.startPipe); p.setBindGroup(0, this.bg); p.dispatchWorkgroups(Math.ceil(T / 64)); p.end();
            d.queue.submit([enc.finish()]);
            return this._batch();
        }
        const base = this.base, round0 = this.round;
        this.round += Rn;
        d.queue.writeBuffer(this.params, 0, this._params(base, round0));
        d.queue.writeBuffer(this.out, 0, new Uint32Array(4));
        const staging = d.createBuffer({ size: this.outBytes, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
        const p = enc.beginComputePass();
        p.setPipeline(this.walkPipe); p.setBindGroup(0, this.bg); p.dispatchWorkgroups(Math.ceil(T / 64)); p.end();
        enc.copyBufferToBuffer(this.out, 0, staging, 0, this.outBytes);
        d.queue.submit([enc.finish()]);
        const keys = T * Rn * STEP;
        return staging.mapAsync(GPUMapMode.READ).then(() => {
            const a = new Uint32Array(staging.getMappedRange().slice(0));
            staging.unmap(); staging.destroy();
            const count = Math.min(a[0], MATCH_CAP), records = [];
            for (let s = 0; s < count; s++) {
                const o = 4 + s * REC;
                records.push({ t: a[o], off: a[o + 1] | 0, pat: a[o + 2], words: a.slice(o + 3, o + 10) });
            }
            return { base, round0, keys, count: a[0], records };
        });
    }

    // Recompute the key and its address on the CPU with the reference implementation.
    verify(base, rec) {
        const k = R.keyOf(this.key, base + BigInt(rec.t), rec.off);
        const bytes = R.addressBytes(R.mulG(k));
        const gb = wordsToAddr(rec.words);
        if (!bytes.every((b, i) => b === gb[i])) return null;
        const address = R.b58encode(bytes);
        if (!R.matches(address, this.pats)) return null;
        return { address, secret: R.keyHex(k) };
    }

    // Runs until stop(); onMatch(match), onStats({keys, rate}).
    async run({ onMatch, onStats, onError, inflight = 3 }) {
        this.running = true;
        let keys = 0, mismatch = 0;
        const t0 = performance.now();
        let lastT = t0, lastK = 0;
        const queue = [];
        try {
            while (this.running || queue.length) {
                while (this.running && queue.length < inflight) queue.push(this._batch());
                const r = await queue.shift();
                if (this.gpuError) throw new Error(this.gpuError);
                keys += r.keys;
                // each check costs ~1 ms of CPU: very short patterns would freeze the page, so cap it per batch
                for (const rec of r.records.slice(0, VERIFY_CAP)) {
                    if (!this.running) break;
                    const m = this.verify(r.base, rec);
                    if (m) onMatch?.(m); else mismatch++;
                }
                const now = performance.now();
                if (now - lastT > 1000 || !this.running) {
                    const avg = (keys * 1000) / (now - t0);
                    // batches finishing back-to-back at the end would show a fake spike: report the average then
                    const rate = this.running ? ((keys - lastK) * 1000) / (now - lastT) : avg;
                    // wall time per batch bounds its GPU time from above: keep batches well under the driver watchdog
                    if (this.running && !this.dumpMode && ((this.T * this.R * STEP) / rate) * 1000 < 100) {
                        if (this.T < this.capT) { this.T *= 2; this.round = ROUNDS; }   // more threads need new starts
                        else if (this.R < 16) this.R *= 2;
                    }
                    onStats?.({ keys, rate, avg, mismatch, dropped: r.count > VERIFY_CAP });
                    lastT = now; lastK = keys;
                }
            }
        } catch (e) { onError?.(e); }
        this.running = false;
    }
    stop() { this.running = false; }

    async readDump() {
        const d = this.device, size = this.dumpBytes;
        const st = d.createBuffer({ size, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
        const enc = d.createCommandEncoder();
        enc.copyBufferToBuffer(this.dump, 0, st, 0, size);
        d.queue.submit([enc.finish()]);
        await st.mapAsync(GPUMapMode.READ);
        const a = new Uint32Array(st.getMappedRange().slice(0));
        st.unmap(); st.destroy();
        return a;
    }
    destroy() { this.device.destroy(); }
}

// 7 words (0x41, hash as 5 big-endian words, checksum) -> 25 bytes
function wordsToAddr(w) {
    const b = new Uint8Array(25);
    b[0] = w[0] & 0xff;
    for (let i = 0; i < 6; i++) for (let q = 0; q < 4; q++) b[1 + 4 * i + q] = (w[1 + i] >>> (24 - 8 * q)) & 0xff;
    return b;
}

// GPU addresses vs reference, GPU matcher vs a full scan, checksum-skip residues. Returns { ok, lines }.
export async function selftest({ gpu } = {}) {
    const lines = [];
    let ok = true;
    const want = "TMVQGm1qAQYVdetCeGRRkTWYYrLXuHK2HC";
    const got = R.addressOfKey(1n);
    lines.push(`${got === want ? "ok  " : "FAIL"} reference: key 1 -> ${got}`); ok &&= got === want;

    const T = 256, Rn = 2, n = T * Rn * STEP;
    const g = await Grinder.create({ gpu, threads: T, maxThreads: T, rounds: Rn, dump: true });
    const sets = [
        [{ suffix: "Ab" }, { prefix: "TRw" }, { prefix: "Tx", suffix: "z" }, { prefix: "TSw" }],
        [{ suffix: "Ab" }, { prefix: "TRw" }, { prefix: "Tx", suffix: "z" }, { prefix: "TSw" }],
        [{ suffix: "tron42x" }, { prefix: "TRw" }, { prefix: "TKe" }],   // suffix of 7: checksum skip on
    ];
    let bad = 0, checked = 0, total = 0;
    for (const set of sets) {   // rounds 0-1 from fresh starts, then from moved centres
        const pats = g.setPatterns(set, true);
        const r = await g._batch();
        const dump = await g.readDump();
        const bytesAt = (i) => wordsToAddr(dump.subarray(i * 9, i * 9 + 7));
        const addrAt = (i) => R.b58encode(bytesAt(i));
        const keyAt = (i) => {
            const t = Math.floor(i / (Rn * STEP)), local = i % (Rn * STEP);
            const rr = Math.floor(local / STEP), o = (local % STEP) - M;
            return { t, off: (r.round0 + rr) * STEP + o };
        };
        for (let j = 0; j < 60; j++) {
            const i = j < 3 * STEP ? j : (j * 2654435761) % n;
            const { t, off } = keyAt(i);
            if (R.addressOfKey(R.keyOf(g.key, r.base + BigInt(t), off)) !== addrAt(i)) bad++;
            checked++;
        }
        const wantSet = [];
        for (let i = 0; i < n; i++) if (R.matches(addrAt(i), g.pats)) { const { t, off } = keyAt(i); wantSet.push(`${t}:${off}`); }
        const gotSet = r.records.map((x) => `${x.t}:${x.off}`);
        wantSet.sort(); gotSet.sort();
        const same = wantSet.length === gotSet.length && wantSet.every((v, i) => v === gotSet[i]);
        const ver = r.records.every((rec) => g.verify(r.base, rec));
        lines.push(`${same && ver ? "ok  " : "FAIL"} GPU matcher vs full scan, rounds ${r.round0}-${r.round0 + Rn - 1}${pats.skip ? ", checksum skip" : ""} (${wantSet.length} expected, ${gotSet.length} found${ver ? "" : ", verify failed"})`);
        ok &&= same && ver;
        if (pats.skip) {
            let rb = 0;
            for (let i = 0; i < n; i++) {
                const gr = (BigInt(dump[i * 9 + 8]) << 32n) | BigInt(dump[i * 9 + 7]);
                if (gr !== R.skipResidue(bytesAt(i), pats.skip.m)) rb++;
            }
            lines.push(`${rb ? "FAIL" : "ok  "} checksum-skip residues vs reference (${n} keys, ${rb} bad)`); ok &&= !rb;
        }
        total += n;
    }
    lines.push(`${bad ? "FAIL" : "ok  "} GPU addresses vs reference (${checked} of ${total} checked, ${bad} bad)`); ok &&= !bad;
    g.destroy();
    return { ok, lines };
}

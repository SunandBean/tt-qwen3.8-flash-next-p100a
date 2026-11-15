"""fused GDN core as one tt-metal program (ttnn.generic_op), one head per Tensix core.
Replaces VerifyModel.vgdn from the conv output to y (q/k L2 norms, decay mask, chunk solve, output, rms norm,
z gate): ~60 ttnn ops per layer -> 1. Kernels: tt_qwen38/kernels/gdn_core/{reader,compute,writer}.cpp.

    k = GdnCore(dev, T)
    k(conv, gcol, bcol, S0, z, nw, y, kn, U)      # all fp32 TILE in DRAM; y/kn/U preallocated (trace-safe)
"""
from __future__ import annotations

import math
import os
import struct

import torch
import ttnn

KDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernels", "gdn_core")
NH = 48


def f2u(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", float(x)))[0]


class GdnCore:
    # CB index -> tiles (fp32 4 KB each unless the tensor says otherwise)
    CBS = {0: 4, 1: 4, 2: 4, 3: 1, 4: 1, 5: 16, 6: 4, 7: 4, 8: 4, 9: 4, 10: 1, 11: 4, 12: 4, 13: 1, 14: 1, 15: 1,
           16: 4, 17: 4, 18: 4, 19: 1, 20: 1, 21: 4, 22: 4, 23: 4, 24: 4, 25: 4, 26: 1, 27: 4, 28: 4, 29: 4}

    def __init__(self, dev, T: int, eps: float = 1e-6, eps_l2: float = 1e-6):
        self.dev, self.T = dev, T
        self.stage = 0                                   # ABG debug stage (see compute.cpp)
        g = dev.compute_with_storage_grid_size()
        self.cores = [ttnn.CoreCoord(h % g.x, h // g.x) for h in range(NH)]
        self.grid = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in self.cores])
        self.ct_args = [T, f2u(eps_l2), f2u(1 / math.sqrt(128)), f2u(1.0), f2u(1 / 128), f2u(eps)]
        c = torch.zeros(1, 1, 32, 128)
        c[0, 0, :, 0:32] = torch.tril(torch.ones(32, 32))
        c[0, 0, :, 32:64] = torch.tril(torch.ones(32, 32), -1)
        c[0, 0, :, 64:96] = 1.0
        c[0, 0, :, 96:128] = torch.eye(32)
        self.consts = ttnn.from_torch(c, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev,
                                      memory_config=ttnn.DRAM_MEMORY_CONFIG)

    def _cbs(self, fmt: dict):
        out = []
        for idx, n in self.CBS.items():
            dt = fmt.get(idx, ttnn.float32)
            page = 4096 if dt == ttnn.float32 else 2048
            out.append(ttnn.CBDescriptor(total_size=page * n, core_ranges=self.grid,
                                         format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=idx, data_format=dt,
                                                                                     page_size=page)]))
        return out

    def __call__(self, conv, gcol, bcol, S0, z, nw, y, kn, U, ab=None, dt=None, a=None, gout=None):
        """ABG mode (ab given): alpha/beta/gamma are computed in the kernel from the ab projection [1,1,32,128] fp32 and
        the per-head dt/a values (lists of 48 floats); gamma column tiles go to gout [1,48,32,32] for the commit."""
        abg = ab is not None
        dummy = self.consts
        ins = [conv, gcol if not abg else dummy, bcol if not abg else dummy, S0, self.consts, z, nw,
               ab if abg else dummy]
        outs = [y, kn, U, gout if abg else kn]
        fmt = {0: conv.dtype, 1: conv.dtype, 2: conv.dtype, 5: S0.dtype, 7: z.dtype, 8: nw.dtype,
               16: y.dtype, 17: kn.dtype, 18: U.dtype}
        if not abg:
            fmt.update({3: gcol.dtype, 4: bcol.dtype})
        r_ct, w_ct = [], []
        for t in ins:
            r_ct += ttnn.TensorAccessorArgs(t).get_compile_time_args()
        r_ct.append(int(abg))
        for t in outs:
            w_ct += ttnn.TensorAccessorArgs(t).get_compile_time_args()
        w_ct.append(int(abg))
        r_rt, w_rt, c_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        for h, c in enumerate(self.cores):
            r_rt[c.x][c.y] = [h] + [t.buffer_address() for t in ins]
            w_rt[c.x][c.y] = [h] + [t.buffer_address() for t in outs]
            c_rt[c.x][c.y] = [h // 32, (48 + h) // 32, f2u(dt[h]), f2u(a[h])] if abg else [0, 0, 0, 0]
        cbs = self._cbs(fmt)
        if abg:
            extra = {30: (ttnn.bfloat16, 1), 31: (ttnn.float32, 4), 32: (ttnn.float32, 2), 33: (ttnn.float32, 1),
                     35: (ttnn.float32, 2), 36: (ttnn.float32, 1), 37: (ttnn.float32, 1)}
            for idx, (dtp, n) in extra.items():
                page = 4096 if dtp == ttnn.float32 else 2048
                cbs.append(ttnn.CBDescriptor(total_size=page * n, core_ranges=self.grid, format_descriptors=[
                    ttnn.CBFormatDescriptor(buffer_index=idx, data_format=dtp, page_size=page)]))
        ccfg = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True)
        if abg:
            modes = [ttnn.UnpackToDestMode.Default] * 64
            for i in (31, 35, 36, 37):
                modes[i] = ttnn.UnpackToDestMode.UnpackToDestFp32
            ccfg.unpack_to_dest_mode = ttnn._ttnn.program_descriptor.VectorUnpackToDestMode(modes)
        kernels = [
            ttnn.KernelDescriptor(kernel_source=f"{KDIR}/reader.cpp", core_ranges=self.grid,
                                  compile_time_args=r_ct, runtime_args=r_rt, config=ttnn.ReaderConfigDescriptor()),
            ttnn.KernelDescriptor(kernel_source=f"{KDIR}/writer.cpp", core_ranges=self.grid,
                                  compile_time_args=w_ct, runtime_args=w_rt, config=ttnn.WriterConfigDescriptor()),
            ttnn.KernelDescriptor(kernel_source=f"{KDIR}/compute.cpp", core_ranges=self.grid,
                                  compile_time_args=self.ct_args + [int(abg), self.stage], runtime_args=c_rt, config=ccfg,
                                  defines=[("INP_FLOAT32", "1")]),     # fp32 softplus polynomial (ttnn sets it too)
        ]
        prog = ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)
        return ttnn.generic_op(ins + outs, prog)

class GdnCommit:
    """the commit of the GDN state for an accepted prefix of n rows, one head per core (48 cores), exact fp32
    state update on the SFPU. Kernels: tt_qwen38/kernels/gdn_commit/*.cpp.
        c = GdnCommit(dev); c.write_sel(n)   (host, once per commit, shared by all layers)
        c(S, kn, U, gcol)                    (S updated in place)"""
    KD = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernels", "gdn_commit")
    CBS = {0: 16, 1: 4, 2: 4, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1, 9: 1, 10: 1, 11: 1, 12: 4, 13: 4, 14: 2, 16: 2}
    EXACT = (0, 4, 6, 9, 11, 14)   # CBs read with copy_tile into fp32 dest; NOT matmul operands (they read as 0)

    def __init__(self, dev, debug: int = 0):
        self.dev = dev
        self.debug = debug
        g = dev.compute_with_storage_grid_size()
        self.cores = [ttnn.CoreCoord(h % g.x, h // g.x) for h in range(NH)]
        self.grid = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in self.cores])
        self.sel = ttnn.from_torch(torch.zeros(1, 1, 32, 64), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev,
                                   memory_config=ttnn.DRAM_MEMORY_CONFIG)

    @staticmethod
    def sel_host(n: int) -> torch.Tensor:
        t = torch.zeros(1, 1, 32, 64)
        t[0, 0, :, n - 1] = 1.0                 # SELROW: every row picks row r = n-1
        t[0, 0, :n, 32] = 1.0                   # mask column (column 0 of the second tile)
        return t

    def write_sel(self, n: int):
        ttnn.copy_host_to_device_tensor(ttnn.from_torch(self.sel_host(n), dtype=ttnn.float32,
                                                        layout=ttnn.TILE_LAYOUT), self.sel)

    def __call__(self, S, kn, U, gcol):
        ins = [S, kn, U, gcol, self.sel]
        r_ct = []
        for t in ins:
            r_ct += ttnn.TensorAccessorArgs(t).get_compile_time_args()
        w_ct = ttnn.TensorAccessorArgs(S).get_compile_time_args()
        r_rt, w_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        for h, c in enumerate(self.cores):
            r_rt[c.x][c.y] = [h] + [t.buffer_address() for t in ins]
            w_rt[c.x][c.y] = [h, S.buffer_address()]
        # CB 7 (hi) is bf16: packing gamma into it rounds to bf16, which a TF32 matmul then reads exactly (an FPU
        # datacopy of fp32 is NOT truncated to TF32, so it cannot make the hi part)
        fmt = lambda i: (ttnn.bfloat16, 2048) if i == 7 else (ttnn.float32, 4096)
        cbs = [ttnn.CBDescriptor(total_size=fmt(i)[1] * n, core_ranges=self.grid,
                                 format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=i, data_format=fmt(i)[0],
                                                                             page_size=fmt(i)[1])])
               for i, n in self.CBS.items()]
        ccfg = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True)
        modes = [ttnn.UnpackToDestMode.Default] * 64
        for i in self.EXACT:
            modes[i] = ttnn.UnpackToDestMode.UnpackToDestFp32
        ccfg.unpack_to_dest_mode = ttnn._ttnn.program_descriptor.VectorUnpackToDestMode(modes)
        ks = [ttnn.KernelDescriptor(kernel_source=f"{self.KD}/reader.cpp", core_ranges=self.grid,
                                    compile_time_args=r_ct, runtime_args=r_rt, config=ttnn.ReaderConfigDescriptor()),
              ttnn.KernelDescriptor(kernel_source=f"{self.KD}/writer.cpp", core_ranges=self.grid,
                                    compile_time_args=w_ct, runtime_args=w_rt, config=ttnn.WriterConfigDescriptor()),
              ttnn.KernelDescriptor(kernel_source=f"{self.KD}/compute.cpp", core_ranges=self.grid,
                                    compile_time_args=[self.debug], runtime_args=[], config=ccfg)]
        return ttnn.generic_op(ins, ttnn.ProgramDescriptor(kernels=ks, semaphores=[], cbs=cbs))


class GdnConv:
    """GDN causal conv over the qkv projection, one program (channel tiles spread over the cores):
    E = hist3 + SH3 @ qkv, conv = silu(sum_k (UP_k @ E) * w_k). Kernels: tt_qwen38/kernels/gdn_conv/*.cpp.
        k = GdnConv(dev); k(qkv, hist3, [w0, w1, w2, w3], conv_out, E_out)"""
    KD = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernels", "gdn_conv")

    def __init__(self, dev, ncores: int = 80):
        self.dev = dev
        g = dev.compute_with_storage_grid_size()
        nt = 320
        per = -(-nt // ncores)
        self.work = [(j0, min(per, nt - j0)) for j0 in range(0, nt, per)]
        self.cores = [ttnn.CoreCoord(i % g.x, i // g.x) for i in range(len(self.work))]
        self.grid = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in self.cores])
        c = torch.zeros(1, 1, 32, 160)
        for r in range(29):
            c[0, 0, r + 3, r] = 1.0                          # SH3: E row r+3 <- qkv row r
        for k in range(4):
            for r in range(32 - k):
                c[0, 0, r, 32 * (k + 1) + r + k] = 1.0       # UP_k: row r <- E row r+k
        self.consts = ttnn.from_torch(c, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev,
                                      memory_config=ttnn.DRAM_MEMORY_CONFIG)

    def __call__(self, qkv, hist, ws, conv, E):
        ins = [qkv, hist, self.consts] + list(ws)
        r_ct, w_ct = [], []
        for t in ins:
            r_ct += ttnn.TensorAccessorArgs(t).get_compile_time_args()
        for t in (conv, E):
            w_ct += ttnn.TensorAccessorArgs(t).get_compile_time_args()
        r_rt, w_rt, c_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        for (j0, n), cc in zip(self.work, self.cores):
            r_rt[cc.x][cc.y] = [j0, n] + [t.buffer_address() for t in ins]
            w_rt[cc.x][cc.y] = [j0, n, conv.buffer_address(), E.buffer_address()]
            c_rt[cc.x][cc.y] = [n]
        sizes = {0: 2, 1: 2, 2: 5, 3: 8, 4: 1, 5: 1, 6: 4, 7: 1, 16: 2, 17: 2}
        cbs = [ttnn.CBDescriptor(total_size=4096 * n, core_ranges=self.grid, format_descriptors=[
            ttnn.CBFormatDescriptor(buffer_index=i, data_format=ttnn.float32, page_size=4096)]) for i, n in sizes.items()]
        ccfg = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True)
        modes = [ttnn.UnpackToDestMode.Default] * 64
        for i in (1, 4, 6):
            modes[i] = ttnn.UnpackToDestMode.UnpackToDestFp32
        ccfg.unpack_to_dest_mode = ttnn._ttnn.program_descriptor.VectorUnpackToDestMode(modes)
        ks = [ttnn.KernelDescriptor(kernel_source=f"{self.KD}/reader.cpp", core_ranges=self.grid,
                                    compile_time_args=r_ct, runtime_args=r_rt, config=ttnn.ReaderConfigDescriptor()),
              ttnn.KernelDescriptor(kernel_source=f"{self.KD}/writer.cpp", core_ranges=self.grid,
                                    compile_time_args=w_ct, runtime_args=w_rt, config=ttnn.WriterConfigDescriptor()),
              ttnn.KernelDescriptor(kernel_source=f"{self.KD}/compute.cpp", core_ranges=self.grid,
                                    compile_time_args=[], runtime_args=c_rt, config=ccfg)]
        return ttnn.generic_op(ins + [conv, E], ttnn.ProgramDescriptor(kernels=ks, semaphores=[], cbs=cbs))


def commit_reference(S0, kn, U, gam, n):
    """float64: S0 [48,128,128], kn/U [48,32,128], gam [32,48] -> S_n."""
    d = torch.float64
    g = gam.to(d).T                                              # [48, 32]
    r = n - 1
    dcol = torch.exp(g[:, r:r + 1] - g) * (torch.arange(32) < n)  # [48, 32]
    return torch.exp(g[:, r])[:, None, None] * S0.to(d) + kn.to(d).transpose(1, 2) @ (U.to(d) * dcol[:, :, None])


def reference(conv, gam, beta, S0, z, nw, T, eps=1e-6, eps_l2=1e-6):
    """float64 torch version of the same math (inputs: conv [32,10240], gam/beta [32,48], S0 [48,128,128],
    z [32,6144], nw [128]). Returns y [32,6144], kn [48,32,128], U [48,32,128] (rows >= T: don't care)."""
    d = torch.float64
    hd = conv.to(d).reshape(32, 80, 128).permute(1, 0, 2)
    q = hd[:16].repeat(3, 1, 1)
    k = hd[16:32].repeat(3, 1, 1)
    v = hd[32:80]
    q = q / torch.sqrt((q * q).sum(-1, keepdim=True) + eps_l2) / math.sqrt(128)
    k = k / torch.sqrt((k * k).sum(-1, keepdim=True) + eps_l2)
    g = gam.to(d).T[:, :, None]                       # [48,32,1]
    b = beta.to(d).T[:, :, None]
    tril = torch.tril(torch.ones(32, 32, dtype=d))
    D = torch.exp((g - g.transpose(1, 2)) * tril) * tril
    A = (k @ k.transpose(1, 2)) * D * torch.tril(torch.ones(32, 32, dtype=d), -1) * b
    eG = torch.exp(g)
    S = S0.to(d)
    VB = (v - (k @ S) * eG) * b
    U = VB.clone()
    for _ in range(T - 1):
        U = VB - A @ U
    O = (q @ S) * eG + ((q @ k.transpose(1, 2)) * D) @ U
    O = O / torch.sqrt((O * O).mean(-1, keepdim=True) + eps) * nw.to(d)
    zz = z.to(d).reshape(32, 48, 128).permute(1, 0, 2)
    y = (O * torch.sigmoid(zz)).permute(1, 0, 2).reshape(32, 6144)
    return y, k, U


def pcc(a, b):
    a, b = a.double().flatten(), b.double().flatten()
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1])


if __name__ == "__main__":                      # python -m tt_qwen38.kgdn [T]
    import tt_qwen38.wallclock  # noqa: F401
    import sys
    import time
    T = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    dev = ttnn.open_device(device_id=0, l1_small_size=24576, trace_region_size=64 * 1024 * 1024)
    try:
        torch.manual_seed(0)
        rows = torch.arange(32)[:, None] < T
        conv = torch.randn(32, 10240) * rows
        g_step = -torch.rand(32, 48) * 0.5 * rows
        gam = torch.cumsum(g_step, 0)
        beta = torch.sigmoid(torch.randn(32, 48)) * rows
        S0 = torch.randn(48, 128, 128) * 0.05
        z = torch.randn(32, 6144)
        nw = 1 + 0.1 * torch.randn(128)
        f = lambda t, shape: ttnn.from_torch(t.reshape(shape).float(), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT,
                                             device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        conv_t = f(conv, (1, 1, 32, 10240))
        gcol = f(gam.T.reshape(48, 32, 1), (1, 48, 32, 1))
        bcol = f(beta.T.reshape(48, 32, 1), (1, 48, 32, 1))
        S_t = f(S0, (1, 48, 128, 128))
        z_t = f(z, (1, 1, 32, 6144))
        nw_t = f(nw, (1, 1, 1, 128))
        y_t = f(torch.zeros(32, 6144), (1, 1, 32, 6144))
        kn_t = f(torch.zeros(48, 32, 128), (1, 48, 32, 128))
        U_t = f(torch.zeros(48, 32, 128), (1, 48, 32, 128))
        k = GdnCore(dev, T)
        t0 = time.perf_counter()
        k(conv_t, gcol, bcol, S_t, z_t, nw_t, y_t, kn_t, U_t)
        ttnn.synchronize_device(dev)
        print(f"first call (compile) {time.perf_counter() - t0:.1f} s", flush=True)
        yr, kr, Ur = reference(conv, gam, beta, S0, z, nw, T)
        yd = ttnn.to_torch(y_t).reshape(32, 6144)
        kd = ttnn.to_torch(kn_t).reshape(48, 32, 128)
        Ud = ttnn.to_torch(U_t).reshape(48, 32, 128)
        for name, a, b in (("y", yd[:T], yr[:T]), ("kn", kd[:, :T], kr[:, :T]), ("U", Ud[:, :T], Ur[:, :T])):
            rel = float((a.double() - b).norm() / b.norm())
            print(f"{name}: PCC {pcc(a, b):.6f}  rel.err {rel:.2e}", flush=True)
        # ABG mode: alpha/beta/gamma from an ab projection inside the kernel
        dtv = (torch.randn(48) * 0.5).tolist()
        av = (-torch.rand(48) * 2 - 0.1).tolist()
        abm = torch.zeros(32, 128)
        abm[:T, :96] = torch.randn(T, 96)
        g_step2 = torch.nn.functional.softplus(abm[:, :48].double() + torch.tensor(dtv, dtype=torch.float64)) * torch.tensor(av, dtype=torch.float64)
        gam2 = torch.cumsum(g_step2, 0)
        beta2 = torch.sigmoid(abm[:, 48:96].double())
        ab_t = f(abm, (1, 1, 32, 128))
        gout = f(torch.zeros(48, 32, 32), (1, 48, 32, 32))
        y3, kn3, U3 = (f(torch.zeros(32, 6144), (1, 1, 32, 6144)), f(torch.zeros(48, 32, 128), (1, 48, 32, 128)),
                       f(torch.zeros(48, 32, 128), (1, 48, 32, 128)))
        yr3, _, Ur3 = reference(conv, gam2.float(), beta2.float(), S0, z, nw, T)
        for st in (1, 2):
            k.stage = st
            k(conv_t, None, None, S_t, z_t, nw_t, y3, kn3, U3, ab=ab_t, dt=dtv, a=av, gout=gout)
            gs = ttnn.to_torch(gout).double().reshape(48, 32, 32)[:, :, 0].T
            want = abm[:, :48].double() if st == 1 else g_step2
            print(f"  ABG stage {st} ({'alpha' if st == 1 else 'g'}): max abs err rows<T {float((gs[:T] - want[:T]).abs().max()):.2e}"
                  f" head0 dev {[round(float(v), 4) for v in gs[:T, 0]]} want {[round(float(v), 4) for v in want[:T, 0]]}", flush=True)
        k.stage = 0
        k(conv_t, None, None, S_t, z_t, nw_t, y3, kn3, U3, ab=ab_t, dt=dtv, a=av, gout=gout)
        gd = ttnn.to_torch(gout).double().reshape(48, 32, 32)[:, :, 0].T
        # the same g through ttnn.softplus (the op the unfused path uses), for the accuracy comparison
        g_tt = ttnn.to_torch(ttnn.multiply(ttnn.softplus(ttnn.add(f(abm[:, :48], (1, 1, 32, 48)), f(torch.tensor(dtv), (1, 1, 1, 48)))),
                                           f(torch.tensor(av), (1, 1, 1, 48)))).double().reshape(32, 48)
        print(f"  ttnn.softplus path: g max abs err rows<T {float((g_tt[:T] - g_step2[:T]).abs().max()):.2e}", flush=True)
        for hh in (0, 5, 40):
            print(f"  ABG head {hh}: gamma dev {[round(float(v), 4) for v in gd[:T, hh]]} ref {[round(float(v), 4) for v in gam2[:T, hh]]}"
                  f" | g_step ref {[round(float(v), 4) for v in g_step2[:T, hh]]} alpha {[round(float(v), 4) for v in abm[:T, hh]]}"
                  f" dt {dtv[hh]:.4f} a {av[hh]:.4f}", flush=True)
        print(f"ABG: gamma max rel err {float(((gd[:T] - gam2[:T]).abs() / gam2[:T].abs()).max()):.2e} | "
              f"y PCC {pcc(ttnn.to_torch(y3).reshape(32, 6144)[:T], yr3[:T]):.6f} | "
              f"U PCC {pcc(ttnn.to_torch(U3).reshape(48, 32, 128)[:, :T], Ur3[:, :T]):.6f}", flush=True)
        # second call with other buffers: runtime args must follow (program cache hit)
        y2 = f(torch.zeros(32, 6144), (1, 1, 32, 6144))
        z2 = f(-z, (1, 1, 32, 6144))
        k(conv_t, gcol, bcol, S_t, z2, nw_t, y2, kn_t, U_t)
        yr2, _, _ = reference(conv, gam, beta, S0, -z, nw, T)
        print(f"cache-hit call, new z/y buffers: PCC {pcc(ttnn.to_torch(y2).reshape(32, 6144)[:T], yr2[:T]):.6f}",
              flush=True)
        # timing: eager loop and one trace of 10 calls
        for _ in range(3):
            k(conv_t, gcol, bcol, S_t, z_t, nw_t, y_t, kn_t, U_t)
        ttnn.synchronize_device(dev)
        tid = ttnn.begin_trace_capture(dev, cq_id=0)
        for _ in range(10):
            k(conv_t, gcol, bcol, S_t, z_t, nw_t, y_t, kn_t, U_t)
        ttnn.end_trace_capture(dev, tid, cq_id=0)
        ttnn.execute_trace(dev, tid, cq_id=0, blocking=True)
        t0 = time.perf_counter()
        for _ in range(5):
            ttnn.execute_trace(dev, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(dev)
        dt = (time.perf_counter() - t0) / 50
        print(f"fused GDN core in a trace: {dt * 1e3:.3f} ms per call", flush=True)
        print(f"after trace: y PCC {pcc(ttnn.to_torch(y_t).reshape(32, 6144)[:T], yr[:T]):.6f}", flush=True)
        ttnn.release_trace(dev, tid)
        # ---- commit kernel: exactness of the state update (large S0 entries; a TF32 multiply would show ~1e-4 rel.)
        kc = GdnCommit(dev)
        for n in (1, 2, T):
            S0c = torch.randn(48, 128, 128) * 100 + 1000
            S_dev = f(S0c, (1, 48, 128, 128))
            kc.write_sel(n)
            kc(S_dev, kn_t, U_t, gcol)
            ref = commit_reference(S0c, kd, Ud, gam, n)
            got = ttnn.to_torch(S_dev).double().reshape(48, 128, 128)
            inc_ref = ref - S0c.double() * torch.exp(gam.double().T[:, n - 1])[:, None, None]
            # the ttnn op sequence of VerifyModel.vcommit on the same inputs (exact-selection matmuls emulated in fp32)
            ck4 = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True)
            gam_t = f(gam, (1, 1, 32, 48))
            gr = ttnn.slice(gam_t, [0, 0, n - 1, 0], [1, 1, n, 48])
            msk = torch.zeros(1, 1, 32, 48); msk[0, 0, :n] = 1
            dcol = ttnn.multiply(ttnn.exp(ttnn.minimum(ttnn.subtract(gr, gam_t), 0.0)), f(msk, (1, 1, 32, 48)))
            dh = ttnn.permute(dcol, (0, 3, 2, 1))
            Sn = ttnn.matmul(kn_t, ttnn.multiply(U_t, dh), transpose_a=True, dtype=ttnn.float32, compute_kernel_config=ck4)
            Gr = ttnn.permute(ttnn.exp(gr), (0, 3, 1, 2))
            S_ops = ttnn.to_torch(ttnn.add(Sn, ttnn.multiply(f(S0c, (1, 48, 128, 128)), Gr))).double().reshape(48, 128, 128)
            print(f"   ttnn ops: max rel err of S {float(((S_ops - ref).abs() / ref.abs().clamp_min(1)).max()):.2e}"
                  f" | kernel vs ops {float(((got - S_ops).abs() / S_ops.abs().clamp_min(1)).max()):.2e}"
                  f" | exp(g_r) rel err: ttnn {float((ttnn.to_torch(ttnn.exp(gr)).double()[0,0,0] / torch.exp(gam.double()[n-1]) - 1).abs().max()):.2e}",
                  flush=True)
            print(f"commit n={n}: max rel err of S {float(((got - ref).abs() / ref.abs().clamp_min(1)).max()):.2e}"
                  f" | increment PCC {pcc(got - S0c.double() * torch.exp(gam.double().T[:, n - 1])[:, None, None], inc_ref):.6f}",
                  flush=True)
        for mode in (2, 3, 4):
            kx = GdnCommit(dev, debug=mode)
            n = 2
            kx.write_sel(n)
            S0c = torch.randn(48, 128, 128) * 100 + 1000
            S_x = f(S0c, (1, 48, 128, 128))
            kx(S_x, kn_t, U_t, gcol)
            got = ttnn.to_torch(S_x).double().reshape(48, 128, 128)
            want = S0c.double()
            if mode == 3:
                want = want * torch.exp(gam.double().T[:, n - 1])[:, None, None]
            if mode == 4:
                dcol = torch.exp(gam.double().T[:, n - 1:n] - gam.double().T) * (torch.arange(32) < n)
                want = want + kd.double().transpose(1, 2) @ (Ud.double() * dcol[:, :, None])
            print(f"debug mode {mode}: max rel err {float(((got - want).abs() / want.abs()).max()):.2e}", flush=True)
        if os.environ.get("KDEBUG"):
            kd_ = GdnCommit(dev, debug=1)
            n = 2
            kd_.write_sel(n)
            S_dbg = f(torch.zeros(48, 128, 128), (1, 48, 128, 128))
            kd_(S_dbg, kn_t, U_t, gcol)
            t = ttnn.to_torch(S_dbg).double().reshape(48, 4, 32, 4, 32).permute(0, 1, 3, 2, 4).reshape(48, 16, 32, 32)
            h = 5
            g = gam.double()[:, h]
            names = ["hi", "lo", "gr", "d", "Gr", "W0", "W1", "W2", "W3", "kT0", "kT1", "kT2", "kT3", "S0_00", "G", "m"]
            exp_ = {"hi+lo": g, "gr": torch.full((32,), float(g[n - 1])),
                    "d": torch.exp(g[n - 1] - g) * (torch.arange(32) < n), "Gr": torch.full((32,), float(torch.exp(g[n - 1]))),
                    "G": g}
            for i, nm in enumerate(names):
                print(f"dbg {nm}: col0 rows0-5 {[round(float(v), 5) for v in t[h, i, :6, 0]]}  row0 cols0-3 {[round(float(v), 5) for v in t[h, i, 0, :4]]}",
                      flush=True)
            print("expected:", {k_: [round(float(v), 5) for v in v_[:6]] for k_, v_ in exp_.items()}, flush=True)
            print("W0 expected col0 rows0-5", [round(float(v), 5) for v in (Ud[h, :6, 0].double() * exp_["d"][:6])], flush=True)
            print("kT0 expected row0 cols0-3", [round(float(v), 5) for v in kd[h, :4, 0]], flush=True)
        S_dev = f(torch.randn(48, 128, 128), (1, 48, 128, 128))
        tid = ttnn.begin_trace_capture(dev, cq_id=0)
        for _ in range(10):
            kc(S_dev, kn_t, U_t, gcol)
        ttnn.end_trace_capture(dev, tid, cq_id=0)
        ttnn.execute_trace(dev, tid, cq_id=0, blocking=True)
        t0 = time.perf_counter()
        for _ in range(5):
            ttnn.execute_trace(dev, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(dev)
        print(f"fused GDN commit in a trace: {(time.perf_counter() - t0) / 50 * 1e3:.3f} ms per call", flush=True)
        ttnn.release_trace(dev, tid)
        # ---- conv kernel vs the ttnn op sequence of vgdn (and float64)
        kv = GdnConv(dev)
        qkv = torch.randn(32, 10240)
        qkv[T:] = 0
        hist = torch.zeros(32, 10240)
        hist[:3] = torch.randn(3, 10240).bfloat16().float()          # TF32-representable like the real history
        wts = [torch.randn(10240) * 0.3 for _ in range(4)]
        q_t, h_t = f(qkv, (1, 1, 32, 10240)), f(hist, (1, 1, 32, 10240))
        w_t = [f(w, (1, 1, 1, 10240)) for w in wts]
        conv_o, E_o = f(torch.zeros(32, 10240), (1, 1, 32, 10240)), f(torch.zeros(32, 10240), (1, 1, 32, 10240))
        kv(q_t, h_t, w_t, conv_o, E_o)
        qk = qkv.double()
        Eref = hist.double().clone()
        Eref[3:] += qk[:29]
        cref = torch.zeros(32, 10240, dtype=torch.float64)
        for k in range(4):
            sh = torch.zeros(32, 10240, dtype=torch.float64)
            sh[:32 - k] = Eref[k:]
            cref += sh * wts[k].double()
        cref = torch.nn.functional.silu(cref)
        cd, Ed = ttnn.to_torch(conv_o).double().reshape(32, 10240), ttnn.to_torch(E_o).double().reshape(32, 10240)
        print(f"conv kernel: rows<T PCC {pcc(cd[:T], cref[:T]):.7f} rel {float((cd[:T] - cref[:T]).norm() / cref[:T].norm()):.2e}"
              f" | E rows<T+3 max err {float((Ed[:T + 3] - Eref[:T + 3]).abs().max()):.2e}", flush=True)
        tid = ttnn.begin_trace_capture(dev, cq_id=0)
        for _ in range(10):
            kv(q_t, h_t, w_t, conv_o, E_o)
        ttnn.end_trace_capture(dev, tid, cq_id=0)
        ttnn.execute_trace(dev, tid, cq_id=0, blocking=True)
        t0 = time.perf_counter()
        for _ in range(5):
            ttnn.execute_trace(dev, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(dev)
        print(f"GDN conv kernel in a trace: {(time.perf_counter() - t0) / 50 * 1e3:.3f} ms per call", flush=True)
        ttnn.release_trace(dev, tid)
    finally:
        ttnn.close_device(dev)

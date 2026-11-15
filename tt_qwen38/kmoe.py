"""resident-expert MoE as two tt-metal programs whose cost follows the ACTIVE experts (not the stored slots).
A (gate/up): work items (active entry j, column tile c < 20) spread over the cores; each reads x (80 tiles) once,
   then per item the 80 gate + 80 up BFP4 tiles of its column, h = silu(gate) * up * w_row -> Hbuf[j, c].
B (down):   each core owns NQ output tiles of y [32, 2560] and sums H[j, kt] @ D_slot(j)[kt, nt] over j, kt.
The host writes one small uint32 control row per layer and pass (count, slot ids, per-row (entry, weight) pairs).
Kernels: tt_qwen38/kernels/moe/*.cpp.  Replaces DeviceExperts.forward_sp (scanning sparse_matmul: ~13 us per STORED slot).
"""
from __future__ import annotations

import os

import numpy as np
import torch
import ttnn

KDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernels", "moe")
CTL_WORDS, CTL_SLOTS, CTL_ROWS, MAX_ACTIVE = 1024, 2, 258, 256
H, F = 2560, 640
PAGE = {ttnn.bfloat16: 2048, ttnn.float32: 4096, ttnn.bfloat4_b: 576, ttnn.bfloat8_b: 1088, ttnn.uint32: 4096}


def _cb(idx, dt, n, grid, page=None):
    page = page or PAGE[dt]
    return ttnn.CBDescriptor(total_size=page * n, core_ranges=grid,
                             format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=idx, data_format=dt,
                                                                         page_size=page)])


class MoeKernel:
    def __init__(self, dev, experts, nc_a: int = 55, nq: int = 4, fidelity=ttnn.MathFidelity.HiFi2):
        self.dev, self.ex = dev, experts
        g = dev.compute_with_storage_grid_size()
        self.gx, self.gy = g.x, g.y
        self.nc_a = min(nc_a, g.x * g.y)
        self.nq = nq
        self.nc_b = (H // 32) // nq
        self.cores_a = [ttnn.CoreCoord(i % g.x, i // g.x) for i in range(self.nc_a)]
        self.cores_b = [ttnn.CoreCoord(i % g.x, i // g.x) for i in range(self.nc_b)]
        self.grid_a = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in self.cores_a])
        self.grid_b = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in self.cores_b])
        self.fid = fidelity
        self.hbuf = ttnn.from_torch(torch.zeros(1, MAX_ACTIVE, 32, F), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                    device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        self.ctl = {L: ttnn.from_torch(torch.zeros(1, 1, 1, CTL_WORDS, dtype=torch.int32), dtype=ttnn.uint32,
                                       layout=ttnn.ROW_MAJOR_LAYOUT, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                    for L in experts.sets if experts.sets[L]}

    # ---- host side
    @staticmethod
    def control(picks: list) -> tuple[torch.Tensor, int]:
        """picks[r] = [(slot, weight), ...] (resident picks of row r, <= 10). Returns (ctl int32 [1,1,1,1024], count)."""
        ctl = np.zeros(CTL_WORDS, np.uint32)
        ctl[CTL_ROWS:CTL_ROWS + 32 * 20] = 0
        ctl[CTL_ROWS:CTL_ROWS + 32 * 20:2] = 0xFFFFFFFF
        entry = {}
        for r, row in enumerate(picks):
            for i, (s, w) in enumerate(row):
                j = entry.setdefault(int(s), len(entry))
                ctl[CTL_ROWS + r * 20 + 2 * i] = j
                ctl[CTL_ROWS + r * 20 + 2 * i + 1] = np.float32(w).view(np.uint32)
        assert len(entry) <= MAX_ACTIVE
        ctl[0], ctl[1] = len(entry), len(picks)
        for s, j in entry.items():
            ctl[CTL_SLOTS + j] = s
        return torch.from_numpy(ctl.view(np.int32).copy()).reshape(1, 1, 1, CTL_WORDS), len(entry)

    def write_control(self, L, ctl: torch.Tensor):
        ttnn.copy_host_to_device_tensor(ttnn.from_torch(ctl, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT),
                                        self.ctl[L])

    # ---- device side
    def __call__(self, L, x, y):
        """x: bf16 TILE [1,1,32,2560] (rows >= T zero); y: fp32 TILE [1,1,32,2560] (preallocated output)."""
        gu, dn, ctl = self.ex.gu[L], self.ex.dn[L], self.ctl[L]
        # program A
        ins, outs = [x, gu, ctl], [self.hbuf]
        r_ct = []
        for t in ins:
            r_ct += ttnn.TensorAccessorArgs(t).get_compile_time_args()
        w_ct = ttnn.TensorAccessorArgs(self.hbuf).get_compile_time_args()
        r_rt, w_rt, c_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        for i, c in enumerate(self.cores_a):
            r_rt[c.x][c.y] = [i, self.nc_a, x.buffer_address(), gu.buffer_address(), ctl.buffer_address()]
            w_rt[c.x][c.y] = [i, self.nc_a, self.hbuf.buffer_address()]
            c_rt[c.x][c.y] = [i, self.nc_a]
        ga = self.grid_a
        cbs = [_cb(0, ttnn.bfloat16, 80, ga), _cb(1, gu.dtype, 16, ga), _cb(2, gu.dtype, 16, ga),
               _cb(3, ttnn.float32, 2, ga), _cb(4, ttnn.uint32, 1, ga), _cb(5, ttnn.float32, 1, ga),
               _cb(6, ttnn.float32, 1, ga), _cb(7, ttnn.float32, 1, ga), _cb(8, ttnn.float32, 1, ga),
               _cb(16, ttnn.bfloat16, 2, ga)]
        ka = [ttnn.KernelDescriptor(kernel_source=f"{KDIR}/a_reader.cpp", core_ranges=ga, compile_time_args=r_ct,
                                    runtime_args=r_rt, config=ttnn.ReaderConfigDescriptor()),
              ttnn.KernelDescriptor(kernel_source=f"{KDIR}/a_writer.cpp", core_ranges=ga, compile_time_args=w_ct,
                                    runtime_args=w_rt, config=ttnn.WriterConfigDescriptor()),
              ttnn.KernelDescriptor(kernel_source=f"{KDIR}/a_compute.cpp", core_ranges=ga, compile_time_args=[],
                                    runtime_args=c_rt,
                                    config=ttnn.ComputeConfigDescriptor(math_fidelity=self.fid, fp32_dest_acc_en=True))]
        ttnn.generic_op(ins + outs, ttnn.ProgramDescriptor(kernels=ka, semaphores=[], cbs=cbs))
        # program B
        nq = self.nq
        ins, outs = [self.hbuf, dn, ctl], [y]
        r_ct = [nq]
        for t in ins:
            r_ct += ttnn.TensorAccessorArgs(t).get_compile_time_args()
        w_ct = [nq] + ttnn.TensorAccessorArgs(y).get_compile_time_args()
        r_rt, w_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        for i, c in enumerate(self.cores_b):
            r_rt[c.x][c.y] = [i * nq, self.hbuf.buffer_address(), dn.buffer_address(), ctl.buffer_address()]
            w_rt[c.x][c.y] = [i * nq, y.buffer_address()]
        gb = self.grid_b
        cbs = [_cb(0, ttnn.bfloat16, 8, gb), _cb(1, dn.dtype, 8 * nq, gb), _cb(4, ttnn.uint32, 1, gb),
               _cb(16, y.dtype, nq, gb)]
        kb = [ttnn.KernelDescriptor(kernel_source=f"{KDIR}/b_reader.cpp", core_ranges=gb, compile_time_args=r_ct,
                                    runtime_args=r_rt, config=ttnn.ReaderConfigDescriptor()),
              ttnn.KernelDescriptor(kernel_source=f"{KDIR}/b_writer.cpp", core_ranges=gb, compile_time_args=w_ct,
                                    runtime_args=w_rt, config=ttnn.WriterConfigDescriptor()),
              ttnn.KernelDescriptor(kernel_source=f"{KDIR}/b_compute.cpp", core_ranges=gb, compile_time_args=[nq],
                                    runtime_args=[],
                                    config=ttnn.ComputeConfigDescriptor(math_fidelity=self.fid, fp32_dest_acc_en=True))]
        return ttnn.generic_op(ins + outs, ttnn.ProgramDescriptor(kernels=kb, semaphores=[], cbs=cbs))


def pcc(a, b):
    a, b = a.double().flatten(), b.double().flatten()
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1])


if __name__ == "__main__":       # python -m tt_qwen38.kmoe [GB] [LAYER]
    import tt_qwen38.wallclock  # noqa: F401
    import sys
    import time
    from tt_qwen38.experts import DeviceExperts
    gb = float(sys.argv[1]) if len(sys.argv) > 1 else 8
    L = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    dev = ttnn.open_device(device_id=0, l1_small_size=24576, trace_region_size=64 * 1024 * 1024)
    try:
        ex = DeviceExperts(dev, gb, "bfp4", "lotto")
        n = len(ex.sets[L])
        print(f"layer {L}: {n} resident slots", flush=True)
        k = MoeKernel(dev, ex)
        gu_w = ttnn.to_torch(ex.gu[L]).float()[0]          # [n, H, 2F]
        dn_w = ttnn.to_torch(ex.dn[L]).float()[0]          # [n, F, H]
        rng = np.random.default_rng(0)
        y_t = ttnn.from_torch(torch.zeros(1, 1, 32, H), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev,
                              memory_config=ttnn.DRAM_MEMORY_CONFIG)
        for T, per_row in ((1, 3), (4, 3), (28, 4)):
            x = torch.zeros(1, 1, 32, H)
            x[0, 0, :T] = torch.randn(T, H) * 0.5
            x_t = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev,
                                  memory_config=ttnn.DRAM_MEMORY_CONFIG)
            picks = [[(int(s), float(w)) for s, w in zip(rng.choice(n, per_row, replace=False), rng.random(per_row))]
                     for _ in range(T)]
            ctl, count = MoeKernel.control(picks)
            k.write_control(L, ctl)
            k(L, x_t, y_t)
            ttnn.synchronize_device(dev)
            yd = ttnn.to_torch(y_t).float()[0, 0, :T]
            xb = x[0, 0, :T].bfloat16().float()
            ref = torch.zeros(T, H)
            for r in range(T):
                for s, w in picks[r]:
                    gu = xb[r] @ gu_w[s]
                    h = (torch.nn.functional.silu(gu[:F]) * gu[F:] * w).bfloat16().float()
                    ref[r] += h @ dn_w[s]
            rel = float((yd - ref).norm() / ref.norm())
            # the current path (scanning sparse_matmul) for the same picks, for time and agreement
            sp = torch.zeros(1, 1, 1, n)
            ws = torch.zeros(1, 1, 32, 32 * n)
            for r in range(T):
                for s, w in picks[r]:
                    sp[0, 0, 0, s] = 1
                    ws[0, 0, r, 32 * s + r] = w
            sp_t = ttnn.from_torch(sp, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev)
            ws_t = ttnn.from_torch(ws, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
            m = torch.zeros(1, 1, 32, H)
            m[0, 0, :T] = 1
            m_t = ttnn.from_torch(m, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
            old = None
            if T <= 16:
                old = ttnn.to_torch(ex.forward_sp(L, x_t, sp_t, ws_t, mask=m_t)).float()[0, 0, :T]

            def timed(fn, reps=10):
                fn()
                ttnn.synchronize_device(dev)
                tid = ttnn.begin_trace_capture(dev, cq_id=0)
                for _ in range(reps):
                    fn()
                ttnn.end_trace_capture(dev, tid, cq_id=0)
                ttnn.execute_trace(dev, tid, cq_id=0, blocking=True)
                t0 = time.perf_counter()
                for _ in range(3):
                    ttnn.execute_trace(dev, tid, cq_id=0, blocking=False)
                ttnn.synchronize_device(dev)
                ttnn.release_trace(dev, tid)
                return (time.perf_counter() - t0) / (3 * reps) * 1e3
            t_new = timed(lambda: k(L, x_t, y_t))
            msg = f"T={T} active={count}: kernel PCC {pcc(yd, ref):.6f} rel {rel:.2e} | {t_new:.3f} ms"
            if old is not None:
                t_old = timed(lambda: ex.forward_sp(L, x_t, sp_t, ws_t, mask=m_t))
                msg += f" | forward_sp PCC {pcc(old, ref):.6f} {t_old:.3f} ms"
            print(msg, flush=True)
    finally:
        ttnn.close_device(dev)

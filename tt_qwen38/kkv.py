"""KV-cache row writer as one data-movement program. Replaces, per attention layer and pass, the per-row loop
slice -> reshape -> pad -> to_layout -> to_memory_config -> paged_update_cache (2T times for K/V, T times for the
indexer keys). One core per (cache, head, 32-column tile) read-modify-writes the cache tiles that hold the new rows.
Kernel: tt_qwen38/kernels/kvw/writer.cpp.
"""
from __future__ import annotations

import os

import torch
import ttnn

KDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernels", "kvw")


class KvWrite:
    def __init__(self, dev, T: int, B: int, W: int, S: int, ncaches: int = 2):
        self.dev, self.T, self.B, self.WT, self.ST, self.nc = dev, T, B, W // 32, S // 32, ncaches
        g = dev.compute_with_storage_grid_size()
        self.work = [(q, b, c) for q in range(ncaches) for b in range(B) for c in range(self.WT)]
        self.cores = [ttnn.CoreCoord(i % g.x, i // g.x) for i in range(len(self.work))]
        self.grid = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in self.cores])

    def __call__(self, caches: list, srcs: list, cur):
        """caches: [K cache, V cache] (or one), bf16 TILE [B,1,S,W]; srcs: matching row-major bf16 [1,T,B,W];
        cur: int32 row-major [2T] (row r at position cur[2r])."""
        c0, s0 = caches[0], srcs[0]
        c1, s1 = (caches[1], srcs[1]) if self.nc > 1 else (c0, s0)
        ins = [c0, s0, c1, s1, cur]
        ct = [self.T, self.B, self.WT, self.ST, 2]
        for t in ins:
            ct += ttnn.TensorAccessorArgs(t).get_compile_time_args()
        pbytes = 8 * self.T
        rt = ttnn.RuntimeArgs()
        for (q, b, c), cc in zip(self.work, self.cores):
            rt[cc.x][cc.y] = [q, b, c, c0.buffer_address(), s0.buffer_address(), c1.buffer_address(),
                              s1.buffer_address(), cur.buffer_address(), pbytes]
        rowb = self.WT * 64
        cbs = [ttnn.CBDescriptor(total_size=2048, core_ranges=self.grid, format_descriptors=[
                   ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.bfloat16, page_size=2048)]),
               ttnn.CBDescriptor(total_size=rowb, core_ranges=self.grid, format_descriptors=[
                   ttnn.CBFormatDescriptor(buffer_index=1, data_format=ttnn.bfloat16, page_size=rowb)]),
               ttnn.CBDescriptor(total_size=max(64, pbytes), core_ranges=self.grid, format_descriptors=[
                   ttnn.CBFormatDescriptor(buffer_index=2, data_format=ttnn.int32, page_size=max(64, pbytes))])]
        k = ttnn.KernelDescriptor(kernel_source=f"{KDIR}/writer.cpp", core_ranges=self.grid, compile_time_args=ct,
                                  runtime_args=rt, config=ttnn.ReaderConfigDescriptor())
        return ttnn.generic_op(ins, ttnn.ProgramDescriptor(kernels=[k], semaphores=[], cbs=cbs))


if __name__ == "__main__":        # python -m tt_qwen38.kkv
    import tt_qwen38.wallclock  # noqa: F401
    import time
    dev = ttnn.open_device(device_id=0, l1_small_size=24576, trace_region_size=64 * 1024 * 1024)
    try:
        S = 4096
        RM, TILE = ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT
        for T, pos in ((4, 30), (1, 777), (28, 100)):
            torch.manual_seed(T)
            K0, V0 = torch.randn(2, 1, S, 256), torch.randn(2, 1, S, 256)
            kc = ttnn.from_torch(K0, dtype=ttnn.bfloat16, layout=TILE, device=dev)
            vc = ttnn.from_torch(V0, dtype=ttnn.bfloat16, layout=TILE, device=dev)
            kn, vn = torch.randn(1, T, 2, 256), torch.randn(1, T, 2, 256)
            ks = ttnn.from_torch(kn, dtype=ttnn.bfloat16, layout=RM, device=dev)
            vs = ttnn.from_torch(vn, dtype=ttnn.bfloat16, layout=RM, device=dev)
            cur = ttnn.from_torch(torch.tensor([pos + u // 2 for u in range(2 * T)], dtype=torch.int32),
                                  dtype=ttnn.int32, layout=RM, device=dev)
            w = KvWrite(dev, T, 2, 256, S)
            w([kc, vc], [ks, vs], cur)
            Kr, Vr = K0.bfloat16().float().clone(), V0.bfloat16().float().clone()
            for r in range(T):
                Kr[:, 0, pos + r] = kn[0, r].bfloat16().float()
                Vr[:, 0, pos + r] = vn[0, r].bfloat16().float()
            okk = torch.equal(ttnn.to_torch(kc).float(), Kr)
            okv = torch.equal(ttnn.to_torch(vc).float(), Vr)
            # indexer-key flavour: one head, 128 wide
            I0 = torch.randn(1, 1, S, 128)
            ic = ttnn.from_torch(I0, dtype=ttnn.bfloat16, layout=TILE, device=dev)
            inew = torch.randn(1, 1, T, 128)
            isrc = ttnn.from_torch(inew, dtype=ttnn.bfloat16, layout=RM, device=dev)
            wi = KvWrite(dev, T, 1, 128, S, ncaches=1)
            wi([ic], [isrc], cur)
            Ir = I0.bfloat16().float().clone()
            Ir[0, 0, pos:pos + T] = inew[0, 0].bfloat16().float()
            oki = torch.equal(ttnn.to_torch(ic).float(), Ir)

            def timed(fn, reps=20):
                fn()
                ttnn.synchronize_device(dev)
                tid = ttnn.begin_trace_capture(dev, cq_id=0)
                for _ in range(reps):
                    fn()
                ttnn.end_trace_capture(dev, tid, cq_id=0)
                ttnn.execute_trace(dev, tid, cq_id=0, blocking=True)
                t0 = time.perf_counter()
                for _ in range(5):
                    ttnn.execute_trace(dev, tid, cq_id=0, blocking=False)
                ttnn.synchronize_device(dev)
                ttnn.release_trace(dev, tid)
                return (time.perf_counter() - t0) / (5 * reps) * 1e3
            print(f"T={T} pos={pos}: K exact {okk}, V exact {okv}, indexer keys exact {oki} | "
                  f"K+V {timed(lambda: w([kc, vc], [ks, vs], cur)):.4f} ms, keys {timed(lambda: wi([ic], [isrc], cur)):.4f} ms",
                  flush=True)
    finally:
        ttnn.close_device(dev)

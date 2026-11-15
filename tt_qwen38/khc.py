"""hyper-connection write as one tt-metal program (80 cores, one column tile each, all 4 branches):
    out_b = mask * (R_b + y * 2 sigmoid(c[:, b]))
Replaces sigmoid, multiply, permute, slice, repeat, multiply, add, where (+ the copy into self.R) of
VerifyModel._gr_write_v. R and the increment are added on the SFPU from fp32 dest (exact residual add).
Kernels: tt_qwen38/kernels/hcw/*.cpp.
"""
from __future__ import annotations

import os
import struct

import torch
import ttnn

KDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernels", "hcw")
PAGE = {ttnn.float32: 4096, ttnn.bfloat16: 2048}


def _cb(idx, dt, n, grid):
    return ttnn.CBDescriptor(total_size=PAGE[dt] * n, core_ranges=grid,
                             format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=idx, data_format=dt,
                                                                         page_size=PAGE[dt])])


class HcWrite:
    def __init__(self, dev, T: int):
        self.dev = dev
        g = dev.compute_with_storage_grid_size()
        self.cores = [ttnn.CoreCoord(i % g.x, i // g.x) for i in range(80)]
        self.grid = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in self.cores])
        e = torch.zeros(1, 1, 32, 128)
        for b in range(4):
            e[0, 0, b, 32 * b] = 1.0
        self.onehot = ttnn.from_torch(e, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev,
                                      memory_config=ttnn.DRAM_MEMORY_CONFIG)
        m = torch.zeros(1, 1, 32, 32)
        m[0, 0, :T] = 1.0
        self.mask = ttnn.from_torch(m, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev,
                                    memory_config=ttnn.DRAM_MEMORY_CONFIG)
        self.two = struct.unpack("<I", struct.pack("<f", 2.0))[0]

    def __call__(self, R, y, c, out):
        """R [1,4,32,2560] fp32, y [1,1,32,2560] fp32, c [1,1,32,32] fp32 (inject logits in columns 0..3),
        out [1,4,32,2560] fp32 (preallocated; may be the model's persistent residual)."""
        ins = [R, y, c, self.onehot, self.mask]
        r_ct = []
        for t in ins:
            r_ct += ttnn.TensorAccessorArgs(t).get_compile_time_args()
        w_ct = ttnn.TensorAccessorArgs(out).get_compile_time_args()
        r_rt, w_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        for i, cc in enumerate(self.cores):
            r_rt[cc.x][cc.y] = [i] + [t.buffer_address() for t in ins]
            w_rt[cc.x][cc.y] = [i, out.buffer_address()]
        g = self.grid
        cbs = [_cb(0, R.dtype, 2, g), _cb(1, y.dtype, 1, g), _cb(2, c.dtype, 1, g), _cb(3, ttnn.float32, 4, g),
               _cb(4, ttnn.float32, 1, g), _cb(5, ttnn.float32, 4, g), _cb(6, ttnn.float32, 1, g),
               _cb(16, out.dtype, 2, g)]
        ccfg = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True)
        modes = [ttnn.UnpackToDestMode.Default] * 64
        for i in (0, 4, 6):
            modes[i] = ttnn.UnpackToDestMode.UnpackToDestFp32
        ccfg.unpack_to_dest_mode = ttnn._ttnn.program_descriptor.VectorUnpackToDestMode(modes)
        ks = [ttnn.KernelDescriptor(kernel_source=f"{KDIR}/reader.cpp", core_ranges=g, compile_time_args=r_ct,
                                    runtime_args=r_rt, config=ttnn.ReaderConfigDescriptor()),
              ttnn.KernelDescriptor(kernel_source=f"{KDIR}/writer.cpp", core_ranges=g, compile_time_args=w_ct,
                                    runtime_args=w_rt, config=ttnn.WriterConfigDescriptor()),
              ttnn.KernelDescriptor(kernel_source=f"{KDIR}/compute.cpp", core_ranges=g, compile_time_args=[self.two],
                                    runtime_args=[], config=ccfg)]
        return ttnn.generic_op(ins + [out], ttnn.ProgramDescriptor(kernels=ks, semaphores=[], cbs=cbs))


if __name__ == "__main__":        # python -m tt_qwen38.khc
    import tt_qwen38.wallclock  # noqa: F401
    import time
    dev = ttnn.open_device(device_id=0, l1_small_size=24576, trace_region_size=64 * 1024 * 1024)
    try:
        T = 4
        torch.manual_seed(0)
        R = torch.zeros(1, 4, 32, 2560)
        R[:, :, :T] = torch.randn(1, 4, T, 2560) * 3 + 1e3 * torch.rand(1, 4, T, 1)   # large offsets: exactness test
        y = torch.randn(1, 1, 32, 2560)
        c = torch.randn(1, 1, 32, 32)
        f = lambda t: ttnn.from_torch(t, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev,
                                      memory_config=ttnn.DRAM_MEMORY_CONFIG)
        Rt, yt, ct, out = f(R), f(y), f(c), f(torch.zeros(1, 4, 32, 2560))
        k = HcWrite(dev, T)
        k(Rt, yt, ct, out)
        coef = 2 * torch.sigmoid(c[0, 0, :, :4].double())                      # [32, 4]
        ref = R.double() + y.double() * coef.T[None, :, :, None]
        ref[:, :, T:] = 0
        got = ttnn.to_torch(out).double()
        inc_ref = (ref - R.double())[:, :, :T]
        inc_got = (got - R.double())[:, :, :T]
        print("max |out - ref|", float((got - ref).abs().max()), "| increment rel.err",
              float((inc_got - inc_ref).norm() / inc_ref.norm()), "| padding rows zero:", bool((got[:, :, T:] == 0).all()),
              flush=True)
        # the ttnn op sequence it replaces (fp32 SFPU eltwise)
        def ops():
            cf = ttnn.multiply(ttnn.sigmoid(ct), 2.0)
            cf = ttnn.slice(ttnn.permute(cf, (0, 3, 2, 1)), [0, 0, 0, 0], [1, 4, 32, 1])
            o = ttnn.add(Rt, ttnn.multiply(ttnn.repeat(yt, ttnn.Shape([1, 4, 1, 1])), cf))
            return o
        o_ops = ttnn.to_torch(ops()).double()
        print("ttnn ops: max |out - ref| (valid rows)", float((o_ops - ref)[:, :, :T].abs().max()), flush=True)

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
        print(f"kernel {timed(lambda: k(Rt, yt, ct, out)):.4f} ms | ttnn ops {timed(ops):.4f} ms", flush=True)
    finally:
        ttnn.close_device(dev)

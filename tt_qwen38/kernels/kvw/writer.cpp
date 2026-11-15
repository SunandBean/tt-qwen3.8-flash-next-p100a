// S2 KV-cache row writer (data movement only). Core (cache q, head b, column tile c) patches the T new rows of head b
// into the bf16 TILE cache [B, 1, S, W]: for each row r at position p = pos[2r] (pos: the int32 cur_pos row of the
// verify pass), read cache tile (b, p/32, c), overwrite row p%32 with src[r, b, 32c .. 32c+31], write it back.
// src is row-major bf16 [1, T, B, W] (page = one (r, b) row of W values). Up to 2 caches (K and V) per launch.
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

constexpr uint32_t cb_t = 0, cb_row = 1, cb_pos = 2;

void kernel_main() {
    constexpr uint32_t T = get_compile_time_arg_val(0);
    constexpr uint32_t B = get_compile_time_arg_val(1);       // heads in the cache (dim 0)
    constexpr uint32_t WT = get_compile_time_arg_val(2);      // tiles per row (W / 32)
    constexpr uint32_t ST = get_compile_time_arg_val(3);      // tile rows per head (S / 32)
    constexpr uint32_t POS_STRIDE = get_compile_time_arg_val(4);  // pos index of row r = r * POS_STRIDE
    constexpr auto c0_a = TensorAccessorArgs<5>();
    constexpr auto s0_a = TensorAccessorArgs<c0_a.next_compile_time_args_offset()>();
    constexpr auto c1_a = TensorAccessorArgs<s0_a.next_compile_time_args_offset()>();
    constexpr auto s1_a = TensorAccessorArgs<c1_a.next_compile_time_args_offset()>();
    constexpr auto p_a = TensorAccessorArgs<s1_a.next_compile_time_args_offset()>();
    const uint32_t q = get_arg_val<uint32_t>(0);          // which cache (0 or 1)
    const uint32_t b = get_arg_val<uint32_t>(1);
    const uint32_t c = get_arg_val<uint32_t>(2);
    const uint32_t c0 = get_arg_val<uint32_t>(3), s0 = get_arg_val<uint32_t>(4);
    const uint32_t c1 = get_arg_val<uint32_t>(5), s1 = get_arg_val<uint32_t>(6);
    const uint32_t p_addr = get_arg_val<uint32_t>(7);
    const uint32_t p_bytes = get_arg_val<uint32_t>(8);
    const uint32_t tb = get_tile_size(cb_t);              // bf16 tile
    const uint32_t rowb = WT * 32 * 2;                    // one src row in bytes
    const auto ca0 = TensorAccessor(c0_a, c0, tb);
    const auto sa0 = TensorAccessor(s0_a, s0, rowb);
    const auto ca1 = TensorAccessor(c1_a, c1, tb);
    const auto sa1 = TensorAccessor(s1_a, s1, rowb);
    const auto pa = TensorAccessor(p_a, p_addr, p_bytes);
    Noc noc;
    CircularBuffer pcb(cb_pos), tcb(cb_t), rcb(cb_row);
    pcb.reserve_back(1);
    noc.async_read(pa, pcb, p_bytes, {.page_id = 0}, {.offset_bytes = 0});
    noc.async_read_barrier();
    volatile int32_t* pos = reinterpret_cast<volatile int32_t*>(pcb.get_write_ptr());
    tcb.reserve_back(1);
    rcb.reserve_back(1);
    volatile uint16_t* tile = reinterpret_cast<volatile uint16_t*>(tcb.get_write_ptr());
    volatile uint16_t* row = reinterpret_cast<volatile uint16_t*>(rcb.get_write_ptr());
    int32_t cur = -1;
    for (uint32_t r = 0; r < T; r++) {
        const int32_t p = pos[r * POS_STRIDE];
        const int32_t tr = p / 32;
        const uint32_t page = (b * ST + tr) * WT + c;
        if (tr != cur) {
            if (cur >= 0) {                                // flush the previous tile
                const uint32_t prev = (b * ST + cur) * WT + c;
                if (q == 0) noc.async_write(use<CircularBuffer::AddrSelector::WRITE_PTR>(tcb), ca0, tb, {.offset_bytes = 0}, {.page_id = prev});
                else noc.async_write(use<CircularBuffer::AddrSelector::WRITE_PTR>(tcb), ca1, tb, {.offset_bytes = 0}, {.page_id = prev});
                noc.async_write_barrier();
            }
            if (q == 0) noc.async_read(ca0, tcb, tb, {.page_id = page}, {.offset_bytes = 0});
            else noc.async_read(ca1, tcb, tb, {.page_id = page}, {.offset_bytes = 0});
            noc.async_read_barrier();
            cur = tr;
        }
        if (q == 0) noc.async_read(sa0, rcb, rowb, {.page_id = r * B + b}, {.offset_bytes = 0});
        else noc.async_read(sa1, rcb, rowb, {.page_id = r * B + b}, {.offset_bytes = 0});
        noc.async_read_barrier();
        const uint32_t rr = p % 32;
        const uint32_t face = (rr / 16) * 2;
        const uint32_t base = face * 256 + (rr % 16) * 16;
        for (uint32_t i = 0; i < 16; i++) {
            tile[base + i] = row[c * 32 + i];
            tile[base + 256 + i] = row[c * 32 + 16 + i];
        }
    }
    if (cur >= 0) {
        const uint32_t prev = (b * ST + cur) * WT + c;
        if (q == 0) noc.async_write(use<CircularBuffer::AddrSelector::WRITE_PTR>(tcb), ca0, tb, {.offset_bytes = 0}, {.page_id = prev});
        else noc.async_write(use<CircularBuffer::AddrSelector::WRITE_PTR>(tcb), ca1, tb, {.offset_bytes = 0}, {.page_id = prev});
        noc.async_write_barrier();
    }
}

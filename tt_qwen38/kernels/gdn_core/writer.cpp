// S2 fused GDN core: writes this head's y tiles into y [1,1,32,6144] (tile columns 4h..4h+3) and the normalized k
// and U tiles into [1,48,32,128] (pages 4h..4h+3) for the commit.
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

constexpr uint32_t cb_y = 16, cb_kno = 17, cb_uo = 18;

void kernel_main() {
    constexpr auto y_a = TensorAccessorArgs<0>();
    constexpr auto k_a = TensorAccessorArgs<y_a.next_compile_time_args_offset()>();
    constexpr auto u_a = TensorAccessorArgs<k_a.next_compile_time_args_offset()>();
    constexpr auto g_a = TensorAccessorArgs<u_a.next_compile_time_args_offset()>();
    constexpr uint32_t ABG = get_compile_time_arg_val(g_a.next_compile_time_args_offset());
    const uint32_t h = get_arg_val<uint32_t>(0);
    const uint32_t y_addr = get_arg_val<uint32_t>(1);
    const uint32_t k_addr = get_arg_val<uint32_t>(2);
    const uint32_t u_addr = get_arg_val<uint32_t>(3);
    const uint32_t g_addr = get_arg_val<uint32_t>(4);
    const uint32_t tb_y = get_tile_size(cb_y);
    const uint32_t tb_k = get_tile_size(cb_kno);
    const uint32_t tb_u = get_tile_size(cb_uo);
    const auto y_acc = TensorAccessor(y_a, y_addr, tb_y);
    const auto k_acc = TensorAccessor(k_a, k_addr, tb_k);
    const auto u_acc = TensorAccessor(u_a, u_addr, tb_u);
    Noc noc;
    auto drain = [&](const auto& acc, uint32_t cb_id, uint32_t base, uint32_t n, uint32_t tb) {
        CircularBuffer cb(cb_id);
        cb.wait_front(n);
        auto src = use<CircularBuffer::AddrSelector::READ_PTR>(cb);
        for (uint32_t t = 0; t < n; t++) {
            noc.async_write(src, acc, tb, {.offset_bytes = t * tb}, {.page_id = base + t});
        }
        noc.async_write_barrier();
        cb.pop_front(n);
    };
    if constexpr (ABG) {                                   // gamma column tile for the commit (not popped)
        constexpr uint32_t cb_g = 3;
        const uint32_t tb_g = get_tile_size(cb_g);
        const auto g_acc = TensorAccessor(g_a, g_addr, tb_g);
        CircularBuffer gcb(cb_g);
        gcb.wait_front(1);
        noc.async_write(use<CircularBuffer::AddrSelector::READ_PTR>(gcb), g_acc, tb_g, {.offset_bytes = 0},
                        {.page_id = h});
        noc.async_write_barrier();
    }
    drain(k_acc, cb_kno, h * 4, 4, tb_k);
    drain(u_acc, cb_uo, h * 4, 4, tb_u);
    drain(y_acc, cb_y, h * 4, 4, tb_y);
}

// S2 hyper-connection write reader: column tile c of y, the inject logits tile, 4 one-hot tiles + the row mask
// (consts), then R_b tile c for b = 0..3 (R is [1,4,32,2560]: page b*80 + c).
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

constexpr uint32_t cb_R = 0, cb_y = 1, cb_c = 2, cb_e = 3, cb_m = 4;

void kernel_main() {
    constexpr auto r_a = TensorAccessorArgs<0>();
    constexpr auto y_a = TensorAccessorArgs<r_a.next_compile_time_args_offset()>();
    constexpr auto c_a = TensorAccessorArgs<y_a.next_compile_time_args_offset()>();
    constexpr auto k_a = TensorAccessorArgs<c_a.next_compile_time_args_offset()>();
    constexpr auto m_a = TensorAccessorArgs<k_a.next_compile_time_args_offset()>();
    const uint32_t c = get_arg_val<uint32_t>(0);
    const uint32_t r_addr = get_arg_val<uint32_t>(1);
    const uint32_t y_addr = get_arg_val<uint32_t>(2);
    const uint32_t c_addr = get_arg_val<uint32_t>(3);
    const uint32_t k_addr = get_arg_val<uint32_t>(4);
    const uint32_t m_addr = get_arg_val<uint32_t>(5);
    const uint32_t tb = get_tile_size(cb_R);
    const auto r_acc = TensorAccessor(r_a, r_addr, tb);
    const auto y_acc = TensorAccessor(y_a, y_addr, get_tile_size(cb_y));
    const auto c_acc = TensorAccessor(c_a, c_addr, get_tile_size(cb_c));
    const auto k_acc = TensorAccessor(k_a, k_addr, get_tile_size(cb_e));
    const auto m_acc = TensorAccessor(m_a, m_addr, get_tile_size(cb_m));
    Noc noc;
    auto read_into = [&](const auto& acc, uint32_t cb_id, uint32_t base, uint32_t n) {
        CircularBuffer cb(cb_id);
        const uint32_t t_b = get_tile_size(cb_id);
        cb.reserve_back(n);
        for (uint32_t t = 0; t < n; t++) {
            noc.async_read(acc, cb, t_b, {.page_id = base + t}, {.offset_bytes = t * t_b});
        }
        noc.async_read_barrier();
        cb.push_back(n);
    };
    read_into(c_acc, cb_c, 0, 1);
    read_into(k_acc, cb_e, 0, 4);
    read_into(m_acc, cb_m, 0, 1);
    read_into(y_acc, cb_y, c, 1);
    for (uint32_t b = 0; b < 4; b++) {
        read_into(r_acc, cb_R, b * 80 + c, 1);
    }
}

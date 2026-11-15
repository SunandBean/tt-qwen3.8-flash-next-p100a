// S2 fused GDN commit reader (head h): gamma column tile (twice: FPU and exact CBs), SELROW + mask tiles (host-written
// per commit), kn and U (4 tiles each, [1,48,32,128]), S0 (16 tiles, [1,48,128,128]).
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

constexpr uint32_t cb_S0 = 0, cb_kn = 1, cb_U = 2, cb_G = 3, cb_Gx = 4, cb_sel = 5, cb_m = 6;

void kernel_main() {
    constexpr auto s_a = TensorAccessorArgs<0>();
    constexpr auto k_a = TensorAccessorArgs<s_a.next_compile_time_args_offset()>();
    constexpr auto u_a = TensorAccessorArgs<k_a.next_compile_time_args_offset()>();
    constexpr auto g_a = TensorAccessorArgs<u_a.next_compile_time_args_offset()>();
    constexpr auto c_a = TensorAccessorArgs<g_a.next_compile_time_args_offset()>();
    const uint32_t h = get_arg_val<uint32_t>(0);
    const uint32_t s_addr = get_arg_val<uint32_t>(1);
    const uint32_t k_addr = get_arg_val<uint32_t>(2);
    const uint32_t u_addr = get_arg_val<uint32_t>(3);
    const uint32_t g_addr = get_arg_val<uint32_t>(4);
    const uint32_t c_addr = get_arg_val<uint32_t>(5);
    const uint32_t tb = get_tile_size(cb_S0);
    const auto s_acc = TensorAccessor(s_a, s_addr, tb);
    const auto k_acc = TensorAccessor(k_a, k_addr, get_tile_size(cb_kn));
    const auto u_acc = TensorAccessor(u_a, u_addr, get_tile_size(cb_U));
    const auto g_acc = TensorAccessor(g_a, g_addr, get_tile_size(cb_G));
    const auto c_acc = TensorAccessor(c_a, c_addr, get_tile_size(cb_sel));
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
    read_into(g_acc, cb_G, h, 1);
    read_into(g_acc, cb_Gx, h, 1);
    read_into(c_acc, cb_sel, 0, 1);
    read_into(c_acc, cb_m, 1, 1);
    read_into(u_acc, cb_U, h * 4, 4);
    read_into(k_acc, cb_kn, h * 4, 4);
    read_into(s_acc, cb_S0, h * 16, 16);
}

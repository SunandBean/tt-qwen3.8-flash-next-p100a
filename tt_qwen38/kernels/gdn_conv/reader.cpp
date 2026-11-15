// S2 GDN conv reader: constants (SH3, UP0..UP3 as 5 tiles) once; per channel tile j of this core: qkv tile j, hist
// tile j and the 4 conv-weight tiles j (weights [1,1,1,10240] x 4, row 0 valid).
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

constexpr uint32_t cb_q = 0, cb_h = 1, cb_c = 2, cb_w = 3;

void kernel_main() {
    constexpr auto q_a = TensorAccessorArgs<0>();
    constexpr auto h_a = TensorAccessorArgs<q_a.next_compile_time_args_offset()>();
    constexpr auto c_a = TensorAccessorArgs<h_a.next_compile_time_args_offset()>();
    constexpr auto w0_a = TensorAccessorArgs<c_a.next_compile_time_args_offset()>();
    constexpr auto w1_a = TensorAccessorArgs<w0_a.next_compile_time_args_offset()>();
    constexpr auto w2_a = TensorAccessorArgs<w1_a.next_compile_time_args_offset()>();
    constexpr auto w3_a = TensorAccessorArgs<w2_a.next_compile_time_args_offset()>();
    const uint32_t j0 = get_arg_val<uint32_t>(0);
    const uint32_t n = get_arg_val<uint32_t>(1);
    const uint32_t q_addr = get_arg_val<uint32_t>(2), h_addr = get_arg_val<uint32_t>(3);
    const uint32_t c_addr = get_arg_val<uint32_t>(4);
    const uint32_t w0 = get_arg_val<uint32_t>(5), w1 = get_arg_val<uint32_t>(6);
    const uint32_t w2 = get_arg_val<uint32_t>(7), w3 = get_arg_val<uint32_t>(8);
    const uint32_t tb = get_tile_size(cb_q);
    const auto q_acc = TensorAccessor(q_a, q_addr, tb);
    const auto h_acc = TensorAccessor(h_a, h_addr, get_tile_size(cb_h));
    const auto c_acc = TensorAccessor(c_a, c_addr, get_tile_size(cb_c));
    const uint32_t tw = get_tile_size(cb_w);
    const auto wa0 = TensorAccessor(w0_a, w0, tw);
    const auto wa1 = TensorAccessor(w1_a, w1, tw);
    const auto wa2 = TensorAccessor(w2_a, w2, tw);
    const auto wa3 = TensorAccessor(w3_a, w3, tw);
    Noc noc;
    CircularBuffer ccb(cb_c);
    ccb.reserve_back(5);
    for (uint32_t t = 0; t < 5; t++) {
        noc.async_read(c_acc, ccb, get_tile_size(cb_c), {.page_id = t}, {.offset_bytes = t * get_tile_size(cb_c)});
    }
    noc.async_read_barrier();
    ccb.push_back(5);
    CircularBuffer qcb(cb_q), hcb(cb_h), wcb(cb_w);
    for (uint32_t i = 0; i < n; i++) {
        const uint32_t j = j0 + i;
        qcb.reserve_back(1);
        hcb.reserve_back(1);
        wcb.reserve_back(4);
        noc.async_read(q_acc, qcb, tb, {.page_id = j}, {.offset_bytes = 0});
        noc.async_read(h_acc, hcb, get_tile_size(cb_h), {.page_id = j}, {.offset_bytes = 0});
        noc.async_read(wa0, wcb, tw, {.page_id = j}, {.offset_bytes = 0});
        noc.async_read(wa1, wcb, tw, {.page_id = j}, {.offset_bytes = tw});
        noc.async_read(wa2, wcb, tw, {.page_id = j}, {.offset_bytes = 2 * tw});
        noc.async_read(wa3, wcb, tw, {.page_id = j}, {.offset_bytes = 3 * tw});
        noc.async_read_barrier();
        qcb.push_back(1);
        hcb.push_back(1);
        wcb.push_back(4);
    }
}

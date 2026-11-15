// Program B reader: this core owns output column tiles nt0..nt0+NQ-1 of y [32, 2560]. For every active entry j and
// k tile kt (F = 20 tiles): H tile (j, kt) and the down-projection tiles D_e[kt, nt0 + q] of slot e = ctl[2 + j].
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

constexpr uint32_t CTL_WORDS = 1024, CTL_SLOTS = 2, FT = 20, NT = 80, KC = 4;
constexpr uint32_t cb_hh = 0, cb_d = 1, cb_ctl = 4;

void kernel_main() {
    constexpr uint32_t NQ = get_compile_time_arg_val(0);
    constexpr auto h_a = TensorAccessorArgs<1>();
    constexpr auto d_a = TensorAccessorArgs<h_a.next_compile_time_args_offset()>();
    constexpr auto c_a = TensorAccessorArgs<d_a.next_compile_time_args_offset()>();
    const uint32_t nt0 = get_arg_val<uint32_t>(0);
    const uint32_t h_addr = get_arg_val<uint32_t>(1);
    const uint32_t d_addr = get_arg_val<uint32_t>(2);
    const uint32_t c_addr = get_arg_val<uint32_t>(3);
    const uint32_t tb_h = get_tile_size(cb_hh);
    const uint32_t tb_d = get_tile_size(cb_d);
    const auto h_acc = TensorAccessor(h_a, h_addr, tb_h);
    const auto d_acc = TensorAccessor(d_a, d_addr, tb_d);
    const auto c_acc = TensorAccessor(c_a, c_addr, CTL_WORDS * 4);
    Noc noc;
    CircularBuffer ccb(cb_ctl);
    ccb.reserve_back(1);
    noc.async_read(c_acc, ccb, CTL_WORDS * 4, {.page_id = 0}, {.offset_bytes = 0});
    noc.async_read_barrier();
    volatile uint32_t* ctl = reinterpret_cast<volatile uint32_t*>(ccb.get_write_ptr());
    ccb.push_back(1);
    const uint32_t count = ctl[0];
    CircularBuffer hcb(cb_hh), dcb(cb_d);
    for (uint32_t j = 0; j < count; j++) {
        const uint32_t base = ctl[CTL_SLOTS + j] * (FT * NT);
        for (uint32_t k0 = 0; k0 < FT; k0 += KC) {
            hcb.reserve_back(KC);
            dcb.reserve_back(KC * NQ);
            for (uint32_t kk = 0; kk < KC; kk++) {
                noc.async_read(h_acc, hcb, tb_h, {.page_id = j * FT + k0 + kk}, {.offset_bytes = kk * tb_h});
                for (uint32_t q = 0; q < NQ; q++) {
                    noc.async_read(d_acc, dcb, tb_d, {.page_id = base + (k0 + kk) * NT + nt0 + q},
                                   {.offset_bytes = (kk * NQ + q) * tb_d});
                }
            }
            noc.async_read_barrier();
            hcb.push_back(KC);
            dcb.push_back(KC * NQ);
        }
    }
}

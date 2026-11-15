// Program A reader: x (80 bf16 tiles) once, then per work item (active entry j, column tile c) the 80 gate tiles of
// column c and the 80 up tiles of column 20 + c of expert slot e = ctl[2 + j], and a routing-weight column tile.
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"
#include <cstdint>
// ---- shared constants (see common.hpp)
constexpr uint32_t CTL_WORDS = 1024;
constexpr uint32_t CTL_SLOTS = 2;
constexpr uint32_t CTL_ROWS = 258;
constexpr uint32_t KT = 80;        // H = 2560
constexpr uint32_t FT = 20;        // F = 640
constexpr uint32_t GU_NT = 40;     // gate_up width 1280
constexpr uint32_t CHUNK = 8;      // weight tiles per CB push

constexpr uint32_t cb_x = 0, cb_g = 1, cb_u = 2, cb_w = 3, cb_ctl = 4;

void kernel_main() {
    constexpr auto x_a = TensorAccessorArgs<0>();
    constexpr auto gu_a = TensorAccessorArgs<x_a.next_compile_time_args_offset()>();
    constexpr auto c_a = TensorAccessorArgs<gu_a.next_compile_time_args_offset()>();
    const uint32_t core = get_arg_val<uint32_t>(0);
    const uint32_t ncores = get_arg_val<uint32_t>(1);
    const uint32_t x_addr = get_arg_val<uint32_t>(2);
    const uint32_t gu_addr = get_arg_val<uint32_t>(3);
    const uint32_t c_addr = get_arg_val<uint32_t>(4);
    const uint32_t tb_x = get_tile_size(cb_x);
    const uint32_t tb_g = get_tile_size(cb_g);
    const auto x_acc = TensorAccessor(x_a, x_addr, tb_x);
    const auto gu_acc = TensorAccessor(gu_a, gu_addr, tb_g);
    const auto c_acc = TensorAccessor(c_a, c_addr, CTL_WORDS * 4);
    Noc noc;

    CircularBuffer ccb(cb_ctl);
    ccb.reserve_back(1);
    noc.async_read(c_acc, ccb, CTL_WORDS * 4, {.page_id = 0}, {.offset_bytes = 0});
    noc.async_read_barrier();
    volatile uint32_t* ctl = reinterpret_cast<volatile uint32_t*>(ccb.get_write_ptr());
    ccb.push_back(1);
    const uint32_t count = ctl[0];
    const uint32_t rows = ctl[1];

    CircularBuffer xcb(cb_x);
    xcb.reserve_back(KT);
    for (uint32_t t = 0; t < KT; t++) {
        noc.async_read(x_acc, xcb, tb_x, {.page_id = t}, {.offset_bytes = t * tb_x});
    }
    noc.async_read_barrier();
    xcb.push_back(KT);

    CircularBuffer gcb(cb_g), ucb(cb_u), wcb(cb_w);
    for (uint32_t i = core; i < count * FT; i += ncores) {
        const uint32_t j = i / FT, c = i % FT;
        const uint32_t base = ctl[CTL_SLOTS + j] * (KT * GU_NT);
        for (uint32_t half = 0; half < 2; half++) {
            CircularBuffer& cb = half ? ucb : gcb;
            const uint32_t col = c + half * FT;
            for (uint32_t k0 = 0; k0 < KT; k0 += CHUNK) {
                cb.reserve_back(CHUNK);
                for (uint32_t t = 0; t < CHUNK; t++) {
                    noc.async_read(gu_acc, cb, tb_g, {.page_id = base + (k0 + t) * GU_NT + col},
                                   {.offset_bytes = t * tb_g});
                }
                noc.async_read_barrier();
                cb.push_back(CHUNK);
            }
        }
        // routing-weight column tile (fp32): element (r, 0) = weight of row r for entry j, 0 elsewhere
        wcb.reserve_back(1);
        volatile uint32_t* w = reinterpret_cast<volatile uint32_t*>(wcb.get_write_ptr());
        for (uint32_t q = 0; q < 1024; q++) w[q] = 0;
        for (uint32_t r = 0; r < rows; r++) {
            for (uint32_t p = 0; p < 10; p++) {
                if (ctl[CTL_ROWS + r * 20 + 2 * p] == j) {
                    const uint32_t off = (r < 16) ? r * 16 : 512 + (r - 16) * 16;
                    w[off] = ctl[CTL_ROWS + r * 20 + 2 * p + 1];
                }
            }
        }
        wcb.push_back(1);
    }
}

// S2 fused GDN core (one head per core): reads this head's tiles into CBs.
// conv [1,1,32,10240] (q 0..63, k 64..127, v 128..319 tile columns; q/k head = h % 16), gcol/bcol [1,48,32,32]
// (column 0 = gamma / beta per row), S0 [1,48,128,128], consts [1,1,32,128] (tril, stril, ones, eye),
// z [1,1,32,6144], nw [1,1,32,128] (row 0 = norm weight).
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

constexpr uint32_t cb_q = 0, cb_k = 1, cb_v = 2, cb_g = 3, cb_b = 4, cb_S = 5, cb_c = 6, cb_z = 7, cb_nw = 8;

void kernel_main() {
    constexpr auto cv_a = TensorAccessorArgs<0>();
    constexpr auto g_a = TensorAccessorArgs<cv_a.next_compile_time_args_offset()>();
    constexpr auto b_a = TensorAccessorArgs<g_a.next_compile_time_args_offset()>();
    constexpr auto s_a = TensorAccessorArgs<b_a.next_compile_time_args_offset()>();
    constexpr auto c_a = TensorAccessorArgs<s_a.next_compile_time_args_offset()>();
    constexpr auto z_a = TensorAccessorArgs<c_a.next_compile_time_args_offset()>();
    constexpr auto n_a = TensorAccessorArgs<z_a.next_compile_time_args_offset()>();
    constexpr auto ab_a = TensorAccessorArgs<n_a.next_compile_time_args_offset()>();
    constexpr uint32_t ABG = get_compile_time_arg_val(ab_a.next_compile_time_args_offset());

    const uint32_t h = get_arg_val<uint32_t>(0);
    const uint32_t cv_addr = get_arg_val<uint32_t>(1);
    const uint32_t g_addr = get_arg_val<uint32_t>(2);
    const uint32_t b_addr = get_arg_val<uint32_t>(3);
    const uint32_t s_addr = get_arg_val<uint32_t>(4);
    const uint32_t c_addr = get_arg_val<uint32_t>(5);
    const uint32_t z_addr = get_arg_val<uint32_t>(6);
    const uint32_t n_addr = get_arg_val<uint32_t>(7);
    const uint32_t ab_addr = get_arg_val<uint32_t>(8);

    const uint32_t tb_cv = get_tile_size(cb_q);
    const uint32_t tb_g = get_tile_size(cb_g);
    const uint32_t tb_b = get_tile_size(cb_b);
    const uint32_t tb_s = get_tile_size(cb_S);
    const uint32_t tb_c = get_tile_size(cb_c);
    const uint32_t tb_z = get_tile_size(cb_z);
    const uint32_t tb_n = get_tile_size(cb_nw);
    const auto cv_acc = TensorAccessor(cv_a, cv_addr, tb_cv);
    const auto g_acc = TensorAccessor(g_a, g_addr, tb_g);
    const auto b_acc = TensorAccessor(b_a, b_addr, tb_b);
    const auto s_acc = TensorAccessor(s_a, s_addr, tb_s);
    const auto c_acc = TensorAccessor(c_a, c_addr, tb_c);
    const auto z_acc = TensorAccessor(z_a, z_addr, tb_z);
    const auto n_acc = TensorAccessor(n_a, n_addr, tb_n);

    Noc noc;
    auto read_into = [&](const auto& acc, uint32_t cb_id, uint32_t base, uint32_t n, uint32_t tb) {
        CircularBuffer cb(cb_id);
        cb.reserve_back(n);
        for (uint32_t t = 0; t < n; t++) {
            noc.async_read(acc, cb, tb, {.page_id = base + t}, {.offset_bytes = t * tb});
        }
        noc.async_read_barrier();
        cb.push_back(n);
    };
    const uint32_t hq = h % 16;
    read_into(c_acc, cb_c, 0, 4, tb_c);
    read_into(cv_acc, cb_q, hq * 4, 4, tb_cv);
    read_into(cv_acc, cb_k, 64 + hq * 4, 4, tb_cv);
    if constexpr (ABG) {
        constexpr uint32_t cb_ab = 31, cb_oh = 32;
        const uint32_t tb_ab = get_tile_size(cb_ab);
        const auto ab_acc = TensorAccessor(ab_a, ab_addr, tb_ab);
        read_into(ab_acc, cb_ab, 0, 4, tb_ab);
        // one-hot tiles (fp32): tile 0 picks column h (alpha), tile 1 column 48 + h (beta) into column 0
        CircularBuffer ohb(cb_oh);
        ohb.reserve_back(2);
        volatile uint32_t* oh = reinterpret_cast<volatile uint32_t*>(ohb.get_write_ptr());
        for (uint32_t q = 0; q < 2048; q++) oh[q] = 0;
        const uint32_t ra = h % 32, rb = (48 + h) % 32;     // row of the one-hot (= source column), column 0
        oh[(ra < 16) ? ra * 16 : 512 + (ra - 16) * 16] = 0x3f800000u;
        oh[1024 + ((rb < 16) ? rb * 16 : 512 + (rb - 16) * 16)] = 0x3f800000u;
        ohb.push_back(2);
    } else {
        read_into(g_acc, cb_g, h, 1, tb_g);
        read_into(b_acc, cb_b, h, 1, tb_b);
    }
    read_into(cv_acc, cb_v, 128 + h * 4, 4, tb_cv);
    read_into(s_acc, cb_S, h * 16, 16, tb_s);
    read_into(z_acc, cb_z, h * 4, 4, tb_z);
    read_into(n_acc, cb_nw, 0, 4, tb_n);
}

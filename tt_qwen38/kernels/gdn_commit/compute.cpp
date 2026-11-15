// S2 fused GDN commit, one head per core: keep the first n rows of the last verify pass.
//   g_r  = gamma[r]  (r = n-1), selected exactly: hi = TF32(gamma) (FPU copy), lo = gamma - hi (SFPU), and
//          SELROW @ hi + SELROW @ lo in fp32 dest (SELROW: every row one-hot at column r)
//   d_i  = exp(mask_i * (g_r - gamma_i)) * mask_i      (mask_i = i < n)
//   S   <- exp(g_r) * S0 + kn^T (U * d)                 (S0 * exp(g_r) and the add on the SFPU: exact fp32 state)
#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/matmul.h"
#include "api/compute/bcast.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/sfpu_binary_bcast.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/transpose.h"
#include "api/compute/reconfig_data_format.h"
#include "api/dataflow/circular_buffer.h"

namespace {
// inputs
constexpr uint32_t cb_S0 = 0, cb_kn = 1, cb_U = 2, cb_G = 3, cb_Gx = 4, cb_sel = 5, cb_m = 6;
// intermediates
constexpr uint32_t cb_hi = 7, cb_lo = 8, cb_gr = 9, cb_d = 10, cb_Gr = 11, cb_W = 12, cb_kT = 13, cb_sn = 14;
constexpr uint32_t cb_out = 16;

inline void WAIT(uint32_t cb, uint32_t n) { CircularBuffer(cb).wait_front(n); }
inline void POP(uint32_t cb, uint32_t n) { CircularBuffer(cb).pop_front(n); }
inline void pack1(uint32_t o) {
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, o, 0);
    tile_regs_release();
    cb_push_back(o, 1);
}
inline void load(uint32_t cb, uint32_t i, uint32_t dst) {     // exact for CBs in UnpackToDestFp32 mode
    reconfig_data_format_srca(cb);
    copy_init(cb);
    copy_tile(cb, i, dst);
}
}  // namespace

void dbg(uint32_t cb, uint32_t i) {           // debug: copy tile i of cb to the output
    cb_reserve_back(cb_out, 1);
    pack_reconfig_data_format(cb_out);
    tile_regs_acquire();
    load(cb, i, 0);
    pack1(cb_out);
}

void kernel_main() {
    constexpr uint32_t DEBUG = get_compile_time_arg_val(0);
    compute_kernel_hw_startup(cb_sel, cb_hi, cb_out);
    WAIT(cb_G, 1);
    WAIT(cb_Gx, 1);
    WAIT(cb_sel, 1);
    WAIT(cb_m, 1);
    // hi = bf16(gamma): packed into the bf16 CB cb_hi (exact in the TF32 matmul below)
    cb_reserve_back(cb_hi, 1);
    pack_reconfig_data_format(cb_hi);
    tile_regs_acquire();
    load(cb_G, 0, 0);
    pack1(cb_hi);
    WAIT(cb_hi, 1);
    // lo = gamma - hi (SFPU, exact)
    cb_reserve_back(cb_lo, 1);
    pack_reconfig_data_format(cb_lo);
    tile_regs_acquire();
    load(cb_Gx, 0, 0);
    load(cb_hi, 0, 1);
    sub_binary_tile_init();
    sub_binary_tile(0, 1, 0);
    pack1(cb_lo);
    WAIT(cb_lo, 1);
    // g_r in column 0 of every row = SELROW @ hi + SELROW @ lo. (A second matmul_init inside one dest acquire drops
    // the first product, so: two matmuls into two tiles, exact SFPU add.)
    for (uint32_t part = 0; part < 2; part++) {
        const uint32_t src = part ? cb_lo : cb_hi;
        cb_reserve_back(cb_sn, 1);
        pack_reconfig_data_format(cb_sn);
        reconfig_data_format(src, cb_sel);
        matmul_init(cb_sel, src, 0);
        tile_regs_acquire();
        matmul_tiles(cb_sel, src, 0, 0, 0);
        pack1(cb_sn);
    }
    WAIT(cb_sn, 2);
    cb_reserve_back(cb_gr, 1);
    pack_reconfig_data_format(cb_gr);
    tile_regs_acquire();
    load(cb_sn, 0, 0);
    load(cb_sn, 1, 1);
    add_binary_tile_init();
    add_binary_tile(0, 1, 0);
    pack1(cb_gr);
    POP(cb_sn, 2);
    WAIT(cb_gr, 1);
    // d = exp(mask * (g_r - gamma)) * mask
    cb_reserve_back(cb_d, 1);
    pack_reconfig_data_format(cb_d);
    tile_regs_acquire();
    load(cb_gr, 0, 0);
    load(cb_Gx, 0, 1);
    sub_binary_tile_init();
    sub_binary_tile(0, 1, 0);
    load(cb_m, 0, 1);
    mul_binary_tile_init();
    mul_binary_tile(0, 1, 0);
    exp_tile_init();
    exp_tile(0);
    mul_binary_tile_init();
    mul_binary_tile(0, 1, 0);
    pack1(cb_d);
    // Gr = exp(g_r) (column 0)
    cb_reserve_back(cb_Gr, 1);
    pack_reconfig_data_format(cb_Gr);
    tile_regs_acquire();
    load(cb_gr, 0, 0);
    exp_tile_init();
    exp_tile(0);
    pack1(cb_Gr);
    WAIT(cb_d, 1);
    WAIT(cb_Gr, 1);
    // W_b = U_b * d (row scale), kT_a = kn_a^T
    WAIT(cb_U, 4);
    WAIT(cb_kn, 4);
    for (uint32_t b = 0; b < 4; b++) {
        cb_reserve_back(cb_W, 1);
        pack_reconfig_data_format(cb_W);
        reconfig_data_format(cb_U, cb_d);
        mul_bcast_cols_init(cb_U, cb_d);
        tile_regs_acquire();
        mul_tiles_bcast_cols(cb_U, cb_d, b, 0, 0);
        pack1(cb_W);
    }
    for (uint32_t a = 0; a < 4; a++) {
        cb_reserve_back(cb_kT, 1);
        pack_reconfig_data_format(cb_kT);
        reconfig_data_format_srca(cb_kn);
        transpose_init(cb_kn);
        tile_regs_acquire();
        transpose_tile(cb_kn, a, 0);
        pack1(cb_kT);
    }
    WAIT(cb_W, 4);
    WAIT(cb_kT, 4);
    WAIT(cb_S0, 16);
    if constexpr (DEBUG >= 2) {           // 2: S <- S0 (exact copy?)  3: S <- S0 * Gr  4: S <- S0 + sn
        for (uint32_t t = 0; t < 16; t++) {
            if constexpr (DEBUG == 4) {
                cb_reserve_back(cb_sn, 1);
                pack_reconfig_data_format(cb_sn);
                reconfig_data_format(cb_W, cb_kT);
                matmul_init(cb_kT, cb_W, 0);
                tile_regs_acquire();
                matmul_tiles(cb_kT, cb_W, t / 4, t % 4, 0);
                pack1(cb_sn);
                WAIT(cb_sn, 1);
            }
            cb_reserve_back(cb_out, 1);
            pack_reconfig_data_format(cb_out);
            tile_regs_acquire();
            load(cb_S0, t, 0);
            if constexpr (DEBUG == 3) {
                load(cb_Gr, 0, 1);
                sfpu_mul_bcast_col_init();
                sfpu_mul_bcast_col(0, 1);
            }
            if constexpr (DEBUG == 4) {
                load(cb_sn, 0, 1);
                add_binary_tile_init();
                add_binary_tile(0, 1, 0);
            }
            pack1(cb_out);
            if constexpr (DEBUG == 4) POP(cb_sn, 1);
        }
        return;
    }
    if constexpr (DEBUG) {
        dbg(cb_hi, 0); dbg(cb_lo, 0); dbg(cb_gr, 0); dbg(cb_d, 0); dbg(cb_Gr, 0);
        for (uint32_t b = 0; b < 4; b++) dbg(cb_W, b);
        for (uint32_t a = 0; a < 4; a++) dbg(cb_kT, a);
        dbg(cb_S0, 0); dbg(cb_G, 0); dbg(cb_m, 0);
        return;
    }
    for (uint32_t a = 0; a < 4; a++) {
        for (uint32_t b = 0; b < 4; b++) {
            // sn = kT_a @ W_b
            cb_reserve_back(cb_sn, 1);
            pack_reconfig_data_format(cb_sn);
            reconfig_data_format(cb_W, cb_kT);
            matmul_init(cb_kT, cb_W, 0);
            tile_regs_acquire();
            matmul_tiles(cb_kT, cb_W, a, b, 0);
            pack1(cb_sn);
            WAIT(cb_sn, 1);
            // S = S0 * Gr + sn (exact)
            cb_reserve_back(cb_out, 1);
            pack_reconfig_data_format(cb_out);
            tile_regs_acquire();
            load(cb_S0, a * 4 + b, 0);
            load(cb_Gr, 0, 1);
            sfpu_mul_bcast_col_init();
            sfpu_mul_bcast_col(0, 1);
            load(cb_sn, 0, 1);
            add_binary_tile_init();
            add_binary_tile(0, 1, 0);
            pack1(cb_out);
            POP(cb_sn, 1);
        }
    }
}

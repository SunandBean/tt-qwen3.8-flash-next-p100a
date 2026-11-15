// S2 GDN causal conv, per channel tile j (32 channels x 32 rows):
//   E_j    = hist_j + SH3 @ qkv_j               (rows 0..2: history, rows 3..: this pass's rows; disjoint -> exact)
//   conv_j = silu( sum_k (UP_k @ E_j) * w_k )   (w_k: row 0 of the conv weight tile, broadcast over rows)
// The sum over the 4 taps is done on the SFPU from fp32 dest (exact adds).
#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/matmul.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/bcast.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/reconfig_data_format.h"
#include "api/dataflow/circular_buffer.h"

namespace {
constexpr uint32_t cb_q = 0, cb_h = 1, cb_c = 2, cb_w = 3, cb_sh = 4, cb_E = 5, cb_t = 6, cb_x = 7;
constexpr uint32_t cb_conv = 16, cb_Eo = 17;

inline void pack1(uint32_t o) {
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, o, 0);
    tile_regs_release();
    cb_push_back(o, 1);
}
}  // namespace

void kernel_main() {
    const uint32_t n = get_arg_val<uint32_t>(0);       // channel tiles of this core
    compute_kernel_hw_startup(cb_c, cb_q, cb_conv);
    CircularBuffer(cb_c).wait_front(5);               // SH3, UP0..UP3
    for (uint32_t i = 0; i < n; i++) {
        CircularBuffer(cb_q).wait_front(1);
        CircularBuffer(cb_h).wait_front(1);
        CircularBuffer(cb_w).wait_front(4);
        // sh = SH3 @ qkv_j
        cb_reserve_back(cb_sh, 1);
        pack_reconfig_data_format(cb_sh);
        reconfig_data_format(cb_q, cb_c);
        matmul_init(cb_c, cb_q, 0);
        tile_regs_acquire();
        matmul_tiles(cb_c, cb_q, 0, 0, 0);
        pack1(cb_sh);
        CircularBuffer(cb_sh).wait_front(1);
        // E = hist + sh (SFPU, exact) -> cb_E (for the taps) and cb_Eo (output, for the commit)
        for (uint32_t o = 0; o < 2; o++) {
            const uint32_t dst = o ? cb_Eo : cb_E;
            cb_reserve_back(dst, 1);
            pack_reconfig_data_format(dst);
            tile_regs_acquire();
            reconfig_data_format_srca(cb_h);
            copy_init(cb_h);
            copy_tile(cb_h, 0, 0);
            reconfig_data_format_srca(cb_sh);
            copy_init(cb_sh);
            copy_tile(cb_sh, 0, 1);
            add_binary_tile_init();
            add_binary_tile(0, 1, 0);
            pack1(dst);
        }
        CircularBuffer(cb_sh).pop_front(1);
        CircularBuffer(cb_q).pop_front(1);
        CircularBuffer(cb_h).pop_front(1);
        CircularBuffer(cb_E).wait_front(1);
        // taps: t_k = (UP_k @ E) * w_k
        for (uint32_t k = 0; k < 4; k++) {
            cb_reserve_back(cb_x, 1);
            pack_reconfig_data_format(cb_x);
            reconfig_data_format(cb_E, cb_c);
            matmul_init(cb_c, cb_E, 0);
            tile_regs_acquire();
            matmul_tiles(cb_c, cb_E, 1 + k, 0, 0);
            pack1(cb_x);
            CircularBuffer(cb_x).wait_front(1);
            cb_reserve_back(cb_t, 1);
            pack_reconfig_data_format(cb_t);
            reconfig_data_format(cb_x, cb_w);
            mul_bcast_rows_init(cb_x, cb_w);
            tile_regs_acquire();
            mul_tiles_bcast_rows(cb_x, cb_w, 0, k, 0);
            pack1(cb_t);
            CircularBuffer(cb_x).pop_front(1);
        }
        CircularBuffer(cb_E).pop_front(1);
        CircularBuffer(cb_w).pop_front(4);
        CircularBuffer(cb_t).wait_front(4);
        // conv = silu(t0 + t1 + t2 + t3)
        cb_reserve_back(cb_conv, 1);
        pack_reconfig_data_format(cb_conv);
        tile_regs_acquire();
        reconfig_data_format_srca(cb_t);
        copy_init(cb_t);
        copy_tile(cb_t, 0, 0);
        for (uint32_t k = 1; k < 4; k++) {
            copy_init(cb_t);
            copy_tile(cb_t, k, 1);
            add_binary_tile_init();
            add_binary_tile(0, 1, 0);
        }
        silu_tile_init();
        silu_tile(0);
        pack1(cb_conv);
        CircularBuffer(cb_t).pop_front(4);
    }
}

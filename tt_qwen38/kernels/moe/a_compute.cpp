// Program A compute: per work item, gate = x @ G[:, c], up = x @ U[:, c] (80 tile matmuls each, bf16 x BFP4 into
// fp32 dest), h = silu(gate) * up * w (w: routing weight per row, column-broadcast) -> one bf16 tile of H.
#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/matmul.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/bcast.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/cb_api.h"
#include "api/dataflow/circular_buffer.h"

namespace {
constexpr uint32_t KT = 80, FT = 20, CHUNK = 8;
constexpr uint32_t cb_x = 0, cb_g = 1, cb_u = 2, cb_w = 3, cb_ctl = 4, cb_ga = 5, cb_ua = 6, cb_s = 7, cb_t = 8;
constexpr uint32_t cb_h = 16;

inline void WAIT(uint32_t cb, uint32_t n) { CircularBuffer(cb).wait_front(n); }
inline void POP(uint32_t cb, uint32_t n) { CircularBuffer(cb).pop_front(n); }

inline void pack1(uint32_t o) {
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, o, 0);
    tile_regs_release();
    cb_push_back(o, 1);
}

void dot(uint32_t w, uint32_t o) {             // o <- x @ (80 weight tiles streamed through cb w)
    cb_reserve_back(o, 1);
    pack_reconfig_data_format(o);
    reconfig_data_format(w, cb_x);
    matmul_init(cb_x, w, 0);
    tile_regs_acquire();
    for (uint32_t k0 = 0; k0 < KT; k0 += CHUNK) {
        WAIT(w, CHUNK);
        for (uint32_t t = 0; t < CHUNK; t++) {
            matmul_tiles(cb_x, w, k0 + t, t, 0);
        }
        POP(w, CHUNK);
    }
    pack1(o);
}
}  // namespace

void kernel_main() {
    const uint32_t core = get_arg_val<uint32_t>(0);
    const uint32_t ncores = get_arg_val<uint32_t>(1);
    compute_kernel_hw_startup(cb_x, cb_g, cb_h);
    WAIT(cb_ctl, 1);
    const uint32_t count = read_tile_value(cb_ctl, 0, 0);
    WAIT(cb_x, KT);
    for (uint32_t i = core; i < count * FT; i += ncores) {
        dot(cb_g, cb_ga);
        dot(cb_u, cb_ua);
        // s = silu(gate)
        WAIT(cb_ga, 1);
        cb_reserve_back(cb_s, 1);
        pack_reconfig_data_format(cb_s);
        reconfig_data_format_srca(cb_ga);
        copy_init(cb_ga);
        tile_regs_acquire();
        copy_tile(cb_ga, 0, 0);
        silu_tile_init();
        silu_tile(0);
        pack1(cb_s);
        POP(cb_ga, 1);
        // t = s * up
        WAIT(cb_s, 1);
        WAIT(cb_ua, 1);
        cb_reserve_back(cb_t, 1);
        pack_reconfig_data_format(cb_t);
        reconfig_data_format(cb_s, cb_ua);
        mul_init(cb_s, cb_ua, false);
        tile_regs_acquire();
        mul_tiles(cb_s, cb_ua, 0, 0, 0);
        pack1(cb_t);
        POP(cb_s, 1);
        POP(cb_ua, 1);
        // h = t * w (column 0 of w broadcast over columns)
        WAIT(cb_t, 1);
        WAIT(cb_w, 1);
        cb_reserve_back(cb_h, 1);
        pack_reconfig_data_format(cb_h);
        reconfig_data_format(cb_t, cb_w);
        mul_bcast_cols_init(cb_t, cb_w);
        tile_regs_acquire();
        mul_tiles_bcast_cols(cb_t, cb_w, 0, 0, 0);
        pack1(cb_h);
        POP(cb_t, 1);
        POP(cb_w, 1);
    }
}

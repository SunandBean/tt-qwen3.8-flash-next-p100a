// S2 hyper-connection write, one output column tile c per core for all 4 branches:
//   coef_b = 2 * sigmoid(c[:, b])   (c = inject logits [32, 32], columns 0..3; column b moved to 0 by E_b)
//   out_b  = mask * (R_b + y * coef_b)
// R, the increment and the mask are unpacked straight to fp32 dest and combined on the SFPU, so the residual
// stream itself is added exactly (no TF32 rounding of R).
#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/matmul.h"
#include "api/compute/bcast.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/reconfig_data_format.h"
#include "api/dataflow/circular_buffer.h"

namespace {
constexpr uint32_t cb_R = 0, cb_y = 1, cb_c = 2, cb_e = 3, cb_m = 4, cb_coef = 5, cb_t = 6, cb_out = 16;

inline void pack1(uint32_t o) {
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, o, 0);
    tile_regs_release();
    cb_push_back(o, 1);
}
}  // namespace

void kernel_main() {
    constexpr uint32_t TWO = get_compile_time_arg_val(0);    // fp32 bits of 2.0
    compute_kernel_hw_startup(cb_c, cb_e, cb_out);
    CircularBuffer(cb_c).wait_front(1);
    CircularBuffer(cb_e).wait_front(4);
    CircularBuffer(cb_m).wait_front(1);
    CircularBuffer(cb_y).wait_front(1);
    // coef_b, column 0 valid
    for (uint32_t b = 0; b < 4; b++) {
        cb_reserve_back(cb_coef, 1);
        pack_reconfig_data_format(cb_coef);
        reconfig_data_format(cb_e, cb_c);
        matmul_init(cb_c, cb_e, 0);
        tile_regs_acquire();
        matmul_tiles(cb_c, cb_e, 0, b, 0);
        sigmoid_tile_init();
        sigmoid_tile(0);
        binop_with_scalar_tile_init();
        mul_unary_tile(0, TWO);
        pack1(cb_coef);
    }
    CircularBuffer(cb_coef).wait_front(4);
    for (uint32_t b = 0; b < 4; b++) {
        // t = y * coef_b (column broadcast)
        cb_reserve_back(cb_t, 1);
        pack_reconfig_data_format(cb_t);
        reconfig_data_format(cb_y, cb_coef);
        mul_bcast_cols_init(cb_y, cb_coef);
        tile_regs_acquire();
        mul_tiles_bcast_cols(cb_y, cb_coef, 0, b, 0);
        pack1(cb_t);
        CircularBuffer(cb_t).wait_front(1);
        CircularBuffer(cb_R).wait_front(1);
        // out = (R + t) * mask, exact fp32 on the SFPU
        cb_reserve_back(cb_out, 1);
        pack_reconfig_data_format(cb_out);
        tile_regs_acquire();
        reconfig_data_format_srca(cb_R);
        copy_init(cb_R);
        copy_tile(cb_R, 0, 0);
        reconfig_data_format_srca(cb_t);
        copy_init(cb_t);
        copy_tile(cb_t, 0, 1);
        add_binary_tile_init();
        add_binary_tile(0, 1, 0);
        reconfig_data_format_srca(cb_m);
        copy_init(cb_m);
        copy_tile(cb_m, 0, 1);
        mul_binary_tile_init();
        mul_binary_tile(0, 1, 0);
        pack1(cb_out);
        CircularBuffer(cb_t).pop_front(1);
        CircularBuffer(cb_R).pop_front(1);
    }
}

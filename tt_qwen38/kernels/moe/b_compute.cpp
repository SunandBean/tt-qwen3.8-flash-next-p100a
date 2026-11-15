// Program B compute: y[:, nt0+q] = sum_j sum_kt H[j, kt] @ D_slot(j)[kt, nt0+q], NQ (<= 4) fp32 dest tiles.
#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/matmul.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/cb_api.h"
#include "api/dataflow/circular_buffer.h"

namespace {
constexpr uint32_t FT = 20, KC = 4;
constexpr uint32_t cb_hh = 0, cb_d = 1, cb_ctl = 4, cb_y = 16;
}  // namespace

void kernel_main() {
    constexpr uint32_t NQ = get_compile_time_arg_val(0);
    compute_kernel_hw_startup(cb_hh, cb_d, cb_y);
    CircularBuffer(cb_ctl).wait_front(1);
    const uint32_t count = read_tile_value(cb_ctl, 0, 0);
    cb_reserve_back(cb_y, NQ);
    pack_reconfig_data_format(cb_y);
    reconfig_data_format(cb_d, cb_hh);
    matmul_init(cb_hh, cb_d, 0);
    tile_regs_acquire();
    for (uint32_t j = 0; j < count; j++) {
        for (uint32_t k0 = 0; k0 < FT; k0 += KC) {
            CircularBuffer(cb_hh).wait_front(KC);
            CircularBuffer(cb_d).wait_front(KC * NQ);
            for (uint32_t kk = 0; kk < KC; kk++) {
                for (uint32_t q = 0; q < NQ; q++) {
                    matmul_tiles(cb_hh, cb_d, kk, kk * NQ + q, q);
                }
            }
            CircularBuffer(cb_hh).pop_front(KC);
            CircularBuffer(cb_d).pop_front(KC * NQ);
        }
    }
    tile_regs_commit();
    tile_regs_wait();
    for (uint32_t q = 0; q < NQ; q++) pack_tile(q, cb_y, q);
    tile_regs_release();
    cb_push_back(cb_y, NQ);
}

// S2 fused GDN core, one head per core (T rows of a 32-row tile; fp32, fp32 dest accumulation).
// Same math as VerifyModel.vgdn from the conv output to y (before the out projection):
//   qn = q * rsqrt(sum q^2 + eps) / sqrt(128) ; kn = k * rsqrt(sum k^2 + eps)
//   D = tril(exp(tril(g_i - g_j)))  (g = cumsum gamma, column 0 of cb_g) ; eG = exp(g)
//   A = stril(kn kn^T * D) * beta_row ; VB = (v - (kn S0) * eG) * beta_row
//   U: U <- VB - A U, T-1 times (exact for strictly lower A) ; O = (qn S0) * eG + ((qn kn^T) * D) U
//   y = rms_norm(O) * w * sigmoid(z)
// Outputs: y (4 tiles), kn (4), U (4) for the commit.
#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/matmul.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/softplus.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/bcast.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/transpose.h"
#include "api/compute/reconfig_data_format.h"
#include "api/dataflow/circular_buffer.h"

namespace {
constexpr uint32_t cb_q = 0, cb_k = 1, cb_v = 2, cb_g = 3, cb_b = 4, cb_S = 5, cb_c = 6, cb_z = 7, cb_nw = 8;
constexpr uint32_t t4a = 9, t1a = 10, cb_qn = 11, cb_kn = 12, t1b = 13, t1c = 14, cb_D = 15;
constexpr uint32_t cb_y = 16, cb_kno = 17, cb_uo = 18;
constexpr uint32_t cb_eG = 19, cb_A = 20, cb_W = 21, cb_VB = 22, cb_U = 23, cb_U2 = 24, cb_QS = 25, cb_P = 26;
constexpr uint32_t cb_O = 27, t4b = 28, cb_sz = 29;
constexpr uint32_t TRIL = 0, STRIL = 1, ONES = 2;
// ABG mode (alpha/beta/gamma from the ab projection): extra CBs
constexpr uint32_t cb_hi16 = 30, cb_ab = 31, cb_oh = 32, cb_lo = 33, cb_prod = 35, cb_col = 36, cb_gv = 37;

inline void WAIT(uint32_t cb, uint32_t n) { CircularBuffer(cb).wait_front(n); }
inline void POP(uint32_t cb, uint32_t n) { CircularBuffer(cb).pop_front(n); }

inline void pack1(uint32_t o) {
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, o, 0);
    tile_regs_release();
    cb_push_back(o, 1);
}

enum { ADD = 0, SUB = 1, MUL = 2 };
// o <- a[ia] (op) b[ib]
void ew1(uint32_t a, uint32_t ia, uint32_t b, uint32_t ib, uint32_t o, int op) {
    cb_reserve_back(o, 1);
    pack_reconfig_data_format(o);
    reconfig_data_format(a, b);
    if (op == ADD) add_init(a, b);
    else if (op == SUB) sub_init(a, b);
    else mul_init(a, b, false);
    tile_regs_acquire();
    if (op == ADD) add_tiles(a, b, ia, ib, 0);
    else if (op == SUB) sub_tiles(a, b, ia, ib, 0);
    else mul_tiles(a, b, ia, ib, 0);
    pack1(o);
}

enum { MUL_COLS = 0, MUL_ROWS = 1, SUB_ROWS = 2 };
// o <- a[ia] (op) broadcast(b[ib]): MUL_COLS uses column 0 of b, *_ROWS row 0 of b
void bc1(uint32_t a, uint32_t ia, uint32_t b, uint32_t ib, uint32_t o, int kind) {
    cb_reserve_back(o, 1);
    pack_reconfig_data_format(o);
    reconfig_data_format(a, b);
    if (kind == MUL_COLS) mul_bcast_cols_init(a, b);
    else if (kind == MUL_ROWS) mul_bcast_rows_init(a, b);
    else sub_bcast_rows_init(a, b);
    tile_regs_acquire();
    if (kind == MUL_COLS) mul_tiles_bcast_cols(a, b, ia, ib, 0);
    else if (kind == MUL_ROWS) mul_tiles_bcast_rows(a, b, ia, ib, 0);
    else sub_tiles_bcast_rows(a, b, ia, ib, 0);
    pack1(o);
}

// o <- sum_{j<n} a[a0 + j*as] @ (tr ? b[b0 + j*bs]^T : b[b0 + j*bs])
void mm1(uint32_t a, uint32_t a0, uint32_t as, uint32_t b, uint32_t b0, uint32_t bs, uint32_t n, bool tr, uint32_t o) {
    cb_reserve_back(o, 1);
    pack_reconfig_data_format(o);
    reconfig_data_format(b, a);
    matmul_init(a, b, tr ? 1 : 0);
    tile_regs_acquire();
    for (uint32_t j = 0; j < n; j++) {
        matmul_tiles(a, b, a0 + j * as, b0 + j * bs, 0);
    }
    pack1(o);
}

// o <- rsqrt(pre * sum_{j<4} a[j] @ ones + eps) * post   (every column of the result holds the row's value)
void rownorm(uint32_t a, uint32_t o, uint32_t pre, uint32_t eps, uint32_t post) {
    cb_reserve_back(o, 1);
    pack_reconfig_data_format(o);
    reconfig_data_format(cb_c, a);
    matmul_init(a, cb_c, 0);
    tile_regs_acquire();
    for (uint32_t j = 0; j < 4; j++) {
        matmul_tiles(a, cb_c, j, ONES, 0);
    }
    binop_with_scalar_tile_init();
    mul_unary_tile(0, pre);
    add_unary_tile(0, eps);
    rsqrt_tile_init();
    rsqrt_tile(0);
    binop_with_scalar_tile_init();
    mul_unary_tile(0, post);
    pack1(o);
}

enum { EXP = 0, SIGMOID = 1, COPY = 2, TRANSPOSE = 3 };
void un1(uint32_t a, uint32_t ia, uint32_t o, int fn) {
    cb_reserve_back(o, 1);
    pack_reconfig_data_format(o);
    reconfig_data_format_srca(a);
    if (fn == TRANSPOSE) {
        transpose_init(a);
        tile_regs_acquire();
        transpose_tile(a, ia, 0);
    } else {
        copy_init(a);
        tile_regs_acquire();
        copy_tile(a, ia, 0);
        if (fn == EXP) {
            exp_tile_init();
            exp_tile(0);
        } else if (fn == SIGMOID) {
            sigmoid_tile_init();
            sigmoid_tile(0);
        }
    }
    pack1(o);
}

inline void loadx(uint32_t cb, uint32_t i, uint32_t dst) {   // copy_tile into dest (exact for UnpackToDestFp32 CBs)
    reconfig_data_format_srca(cb);
    copy_init(cb);
    copy_tile(cb, i, dst);
}

// out (fp32, packed to o) = X @ B computed to ~fp32: X = hi + lo with hi = bf16(X) (exact in the TF32 matmul) and
// lo = X - hi; two matmuls into two tiles, exact SFPU add. Xcb tile xi must be an exact-copy CB (or bf16-exact data).
void mm_exact(uint32_t xcb, uint32_t xi, bool x_left, uint32_t bcb, uint32_t bi, uint32_t o) {
    // hi
    cb_reserve_back(cb_hi16, 1);
    pack_reconfig_data_format(cb_hi16);
    tile_regs_acquire();
    loadx(xcb, xi, 0);
    pack1(cb_hi16);
    CircularBuffer(cb_hi16).wait_front(1);
    // lo
    cb_reserve_back(cb_lo, 1);
    pack_reconfig_data_format(cb_lo);
    tile_regs_acquire();
    loadx(xcb, xi, 0);
    loadx(cb_hi16, 0, 1);
    sub_binary_tile_init();
    sub_binary_tile(0, 1, 0);
    pack1(cb_lo);
    CircularBuffer(cb_lo).wait_front(1);
    for (uint32_t part = 0; part < 2; part++) {
        const uint32_t src = part ? cb_lo : cb_hi16;
        cb_reserve_back(cb_prod, 1);
        pack_reconfig_data_format(cb_prod);
        if (x_left) {
            reconfig_data_format(bcb, src);
            matmul_init(src, bcb, 0);
            tile_regs_acquire();
            matmul_tiles(src, bcb, 0, bi, 0);
        } else {
            reconfig_data_format(src, bcb);
            matmul_init(bcb, src, 0);
            tile_regs_acquire();
            matmul_tiles(bcb, src, bi, 0, 0);
        }
        pack1(cb_prod);
    }
    CircularBuffer(cb_hi16).pop_front(1);
    CircularBuffer(cb_lo).pop_front(1);
    CircularBuffer(cb_prod).wait_front(2);
    cb_reserve_back(o, 1);
    pack_reconfig_data_format(o);
    tile_regs_acquire();
    loadx(cb_prod, 0, 0);
    loadx(cb_prod, 1, 1);
    add_binary_tile_init();
    add_binary_tile(0, 1, 0);
    pack1(o);
    CircularBuffer(cb_prod).pop_front(2);
}
}  // namespace

void kernel_main() {
    constexpr uint32_t T = get_compile_time_arg_val(0);
    constexpr uint32_t EPS_L2 = get_compile_time_arg_val(1);    // fp32 bits: 1e-6
    constexpr uint32_t QSCALE = get_compile_time_arg_val(2);    // 1/sqrt(128)
    constexpr uint32_t ONE = get_compile_time_arg_val(3);       // 1.0
    constexpr uint32_t INV128 = get_compile_time_arg_val(4);    // 1/128
    constexpr uint32_t EPS_RMS = get_compile_time_arg_val(5);   // cfg.eps
    constexpr uint32_t ABG = get_compile_time_arg_val(6);       // 1: alpha/beta/gamma from the ab projection here

    compute_kernel_hw_startup(cb_q, cb_k, cb_y);
    WAIT(cb_c, 4);
    WAIT(cb_q, 4);
    WAIT(cb_k, 4);

    // ---- q, k L2 norms
    for (uint32_t j = 0; j < 4; j++) ew1(cb_q, j, cb_q, j, t4a, MUL);
    WAIT(t4a, 4);
    rownorm(t4a, t1a, ONE, EPS_L2, QSCALE);
    POP(t4a, 4);
    WAIT(t1a, 1);
    for (uint32_t j = 0; j < 4; j++) ew1(cb_q, j, t1a, 0, cb_qn, MUL);
    POP(t1a, 1);
    for (uint32_t j = 0; j < 4; j++) ew1(cb_k, j, cb_k, j, t4a, MUL);
    WAIT(t4a, 4);
    rownorm(t4a, t1a, ONE, EPS_L2, ONE);
    POP(t4a, 4);
    WAIT(t1a, 1);
    for (uint32_t j = 0; j < 4; j++) ew1(cb_k, j, t1a, 0, cb_kn, MUL);
    POP(t1a, 1);
    POP(cb_q, 4);
    POP(cb_k, 4);
    WAIT(cb_qn, 4);
    WAIT(cb_kn, 4);
    for (uint32_t j = 0; j < 4; j++) un1(cb_kn, j, cb_kno, COPY);

    if constexpr (ABG) {
        // alpha_h, beta_h columns of the ab projection (exact), g = softplus(alpha + dt) * a, gamma = cumsum(g),
        // beta = sigmoid(beta)
        const uint32_t ta = get_arg_val<uint32_t>(0), tb = get_arg_val<uint32_t>(1);
        const uint32_t dt = get_arg_val<uint32_t>(2), a = get_arg_val<uint32_t>(3);
        WAIT(cb_ab, 4);
        WAIT(cb_oh, 2);
        constexpr uint32_t STAGE = get_compile_time_arg_val(7);   // debug: 1 gamma := alpha, 2 gamma := g, 0/3 normal
        mm_exact(cb_ab, ta, true, cb_oh, 0, cb_col);           // column 0 = alpha
        CircularBuffer(cb_col).wait_front(1);
        if constexpr (STAGE == 1) {
            cb_reserve_back(cb_g, 1);
            pack_reconfig_data_format(cb_g);
            tile_regs_acquire();
            loadx(cb_col, 0, 0);
            pack1(cb_g);
        }
        // g (fp32, own exact CB: cb_prod must only ever hold aligned pairs, a single tile there makes the next pair
        // wrap and "tile 1" read past the end of the buffer) = softplus(alpha + dt) * a
        cb_reserve_back(cb_gv, 1);
        pack_reconfig_data_format(cb_gv);
        tile_regs_acquire();
        loadx(cb_col, 0, 0);
        binop_with_scalar_tile_init();
        add_unary_tile(0, dt);
        softplus_tile_init();
        softplus_tile(0, ONE, ONE, 0x41a00000u);               // beta 1, threshold 20 (ttnn.softplus defaults)
        binop_with_scalar_tile_init();
        mul_unary_tile(0, a);
        pack1(cb_gv);
        CircularBuffer(cb_col).pop_front(1);
        CircularBuffer(cb_gv).wait_front(1);
        if constexpr (STAGE == 2) {
            cb_reserve_back(cb_g, 1);
            pack_reconfig_data_format(cb_g);
            tile_regs_acquire();
            loadx(cb_gv, 0, 0);
            pack1(cb_g);
        } else if constexpr (STAGE != 1) {
            mm_exact(cb_gv, 0, false, cb_c, TRIL, cb_g);      // gamma = TRIL @ g (exact split)
        }
        CircularBuffer(cb_gv).pop_front(1);
        mm_exact(cb_ab, tb, true, cb_oh, 1, cb_col);           // column 0 = beta logit
        CircularBuffer(cb_col).wait_front(1);
        cb_reserve_back(cb_b, 1);
        pack_reconfig_data_format(cb_b);
        tile_regs_acquire();
        loadx(cb_col, 0, 0);
        sigmoid_tile_init();
        sigmoid_tile(0);
        pack1(cb_b);
        CircularBuffer(cb_col).pop_front(1);
    }
    // ---- D = tril(exp(tril(g_i - g_j))), eG = exp(g)
    WAIT(cb_g, 1);
    WAIT(cb_b, 1);
    bc1(cb_c, ONES, cb_g, 0, t1b, MUL_COLS);       // g_i in every column
    un1(cb_g, 0, t1c, TRANSPOSE);                  // row 0 = g_j
    WAIT(t1b, 1);
    WAIT(t1c, 1);
    bc1(t1b, 0, t1c, 0, t1a, SUB_ROWS);
    POP(t1b, 1);
    POP(t1c, 1);
    WAIT(t1a, 1);
    ew1(t1a, 0, cb_c, TRIL, t1b, MUL);
    POP(t1a, 1);
    WAIT(t1b, 1);
    un1(t1b, 0, t1c, EXP);
    POP(t1b, 1);
    WAIT(t1c, 1);
    ew1(t1c, 0, cb_c, TRIL, cb_D, MUL);
    POP(t1c, 1);
    WAIT(cb_D, 1);
    un1(cb_g, 0, cb_eG, EXP);
    WAIT(cb_eG, 1);

    // ---- A = stril(kn kn^T * D) * beta
    mm1(cb_kn, 0, 1, cb_kn, 0, 1, 4, true, t1a);
    WAIT(t1a, 1);
    ew1(t1a, 0, cb_D, 0, t1b, MUL);
    POP(t1a, 1);
    WAIT(t1b, 1);
    ew1(t1b, 0, cb_c, STRIL, t1c, MUL);
    POP(t1b, 1);
    WAIT(t1c, 1);
    bc1(t1c, 0, cb_b, 0, cb_A, MUL_COLS);
    POP(t1c, 1);
    WAIT(cb_A, 1);

    // ---- VB = (v - (kn S0) * eG) * beta
    WAIT(cb_S, 16);
    for (uint32_t c = 0; c < 4; c++) mm1(cb_kn, 0, 1, cb_S, c, 4, 4, false, t4a);
    WAIT(t4a, 4);
    for (uint32_t c = 0; c < 4; c++) bc1(t4a, c, cb_eG, 0, cb_W, MUL_COLS);
    POP(t4a, 4);
    WAIT(cb_W, 4);
    WAIT(cb_v, 4);
    for (uint32_t c = 0; c < 4; c++) ew1(cb_v, c, cb_W, c, t4a, SUB);
    POP(cb_W, 4);
    POP(cb_v, 4);
    WAIT(t4a, 4);
    for (uint32_t c = 0; c < 4; c++) bc1(t4a, c, cb_b, 0, cb_VB, MUL_COLS);
    POP(t4a, 4);
    WAIT(cb_VB, 4);

    // ---- U: sweep U <- VB - A U (T-1 times)
    uint32_t cur = cb_VB;
    for (uint32_t s = 1; s < T; s++) {
        const uint32_t nxt = (s & 1) ? cb_U : cb_U2;
        for (uint32_t c = 0; c < 4; c++) mm1(cb_A, 0, 0, cur, c, 0, 1, false, t4a);
        WAIT(t4a, 4);
        for (uint32_t c = 0; c < 4; c++) ew1(cb_VB, c, t4a, c, nxt, SUB);
        POP(t4a, 4);
        if (cur != cb_VB) POP(cur, 4);
        WAIT(nxt, 4);
        cur = nxt;
    }
    for (uint32_t c = 0; c < 4; c++) un1(cur, c, cb_uo, COPY);

    // ---- O = (qn S0) * eG + ((qn kn^T) * D) U
    for (uint32_t c = 0; c < 4; c++) mm1(cb_qn, 0, 1, cb_S, c, 4, 4, false, t4a);
    WAIT(t4a, 4);
    for (uint32_t c = 0; c < 4; c++) bc1(t4a, c, cb_eG, 0, cb_QS, MUL_COLS);
    POP(t4a, 4);
    WAIT(cb_QS, 4);
    mm1(cb_qn, 0, 1, cb_kn, 0, 1, 4, true, t1a);
    WAIT(t1a, 1);
    ew1(t1a, 0, cb_D, 0, cb_P, MUL);
    POP(t1a, 1);
    WAIT(cb_P, 1);
    for (uint32_t c = 0; c < 4; c++) mm1(cb_P, 0, 0, cur, c, 0, 1, false, t4a);
    WAIT(t4a, 4);
    for (uint32_t c = 0; c < 4; c++) ew1(cb_QS, c, t4a, c, cb_O, ADD);
    POP(t4a, 4);
    POP(cb_QS, 4);
    WAIT(cb_O, 4);

    // ---- y = rms_norm(O) * w * sigmoid(z)
    for (uint32_t c = 0; c < 4; c++) ew1(cb_O, c, cb_O, c, t4a, MUL);
    WAIT(t4a, 4);
    rownorm(t4a, t1a, INV128, EPS_RMS, ONE);
    POP(t4a, 4);
    WAIT(t1a, 1);
    for (uint32_t c = 0; c < 4; c++) ew1(cb_O, c, t1a, 0, t4b, MUL);
    POP(t1a, 1);
    WAIT(t4b, 4);
    WAIT(cb_nw, 4);
    for (uint32_t c = 0; c < 4; c++) bc1(t4b, c, cb_nw, c, t4a, MUL_ROWS);
    POP(t4b, 4);
    WAIT(t4a, 4);
    WAIT(cb_z, 4);
    for (uint32_t c = 0; c < 4; c++) un1(cb_z, c, cb_sz, SIGMOID);
    WAIT(cb_sz, 4);
    for (uint32_t c = 0; c < 4; c++) ew1(t4a, c, cb_sz, c, cb_y, MUL);
}

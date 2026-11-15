// hostexp: routed-expert FFNs on the host CPU, straight from the GGUF's quantized bytes, with ggml's CPU dot products
// (ggml / llama.cpp, MIT). One call = one token, one layer, any number of experts:
//   y = sum_e w_e * down_e( silu(gate_e x) * up_e x )
// Activations are quantized to each weight type's ggml vec_dot_type (q8_K / q8_0), as llama.cpp and Strata do.
// Threads: a persistent OpenMP team, rows split statically.
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

#include <omp.h>

#include "ggml.h"
#include "ggml-cpu.h"

extern "C" {

int hx_init(int nthreads) {
    ggml_cpu_init();
    if (nthreads > 0) omp_set_num_threads(nthreads);
    return omp_get_max_threads();
}

size_t hx_row_size(int type, int64_t n) { return ggml_row_size((ggml_type) type, n); }

// gate/up/down: per-expert base pointers ([F rows x H] for gate/up, [H rows x F] for down), types are ggml type ids.
// x: [H] float; wts: [n_exp]; y: [H] float (overwritten). Returns 0 on success.
int hx_moe(int n_exp, const void* const* gate, const void* const* up, const void* const* down, int t_gate, int t_up,
           int t_down, const float* wts, const float* x, int H, int F, float* y) {
    const auto* tg = ggml_get_type_traits_cpu((ggml_type) t_gate);
    const auto* tu = ggml_get_type_traits_cpu((ggml_type) t_up);
    const auto* td = ggml_get_type_traits_cpu((ggml_type) t_down);
    if (!tg->vec_dot || !tu->vec_dot || !td->vec_dot || tg->vec_dot_type != tu->vec_dot_type) return 1;
    const ggml_type vx = tg->vec_dot_type, vh = td->vec_dot_type;
    const size_t rb_g = ggml_row_size((ggml_type) t_gate, H), rb_u = ggml_row_size((ggml_type) t_up, H);
    const size_t rb_d = ggml_row_size((ggml_type) t_down, F);
    std::vector<uint8_t> xq(ggml_row_size(vx, H));
    ggml_get_type_traits_cpu(vx)->from_float(x, xq.data(), H);
    std::vector<float> g((size_t) n_exp * F), u((size_t) n_exp * F);
    const int n_gu = n_exp * F;
#pragma omp parallel for schedule(static)
    for (int i = 0; i < 2 * n_gu; ++i) {
        const int j = i % n_gu, e = j / F, r = j % F;
        if (i < n_gu)
            tg->vec_dot(H, &g[j], 0, (const uint8_t*) gate[e] + r * rb_g, 0, xq.data(), 0, 1);
        else
            tu->vec_dot(H, &u[j], 0, (const uint8_t*) up[e] + r * rb_u, 0, xq.data(), 0, 1);
    }
    std::vector<float> h((size_t) n_exp * F);
    for (int j = 0; j < n_gu; ++j) h[j] = g[j] / (1.0f + std::exp(-g[j])) * u[j];
    const size_t hq_rb = ggml_row_size(vh, F);
    std::vector<uint8_t> hq(hq_rb * n_exp);
    for (int e = 0; e < n_exp; ++e) ggml_get_type_traits_cpu(vh)->from_float(&h[(size_t) e * F], hq.data() + e * hq_rb, F);
#pragma omp parallel for schedule(static)
    for (int c = 0; c < H; ++c) {
        float acc = 0.f;
        for (int e = 0; e < n_exp; ++e) {
            float s;
            td->vec_dot(F, &s, 0, (const uint8_t*) down[e] + c * rb_d, 0, hq.data() + e * hq_rb, 0, 1);
            acc += wts[e] * s;
        }
        y[c] = acc;
    }
    return 0;
}

// Several tokens (rows) at once (M8): each unique expert's weight rows are read once and used for every row that
// routed to it. Pairs (row, expert) come grouped by unique expert: pairs u_off[u] .. u_off[u+1]-1 use expert u.
// x: [n_rows, H]; pair_row/pair_w: [n_pairs]; y: [n_rows, H] (overwritten).
int hx_moe_rows(int n_rows, int n_u, const void* const* gate, const void* const* up, const void* const* down,
                int t_gate, int t_up, int t_down, const int* u_off, const int* pair_row, const float* pair_w,
                const float* x, int H, int F, float* y) {
    const auto* tg = ggml_get_type_traits_cpu((ggml_type) t_gate);
    const auto* tu = ggml_get_type_traits_cpu((ggml_type) t_up);
    const auto* td = ggml_get_type_traits_cpu((ggml_type) t_down);
    if (!tg->vec_dot || !tu->vec_dot || !td->vec_dot || tg->vec_dot_type != tu->vec_dot_type) return 1;
    const ggml_type vx = tg->vec_dot_type, vh = td->vec_dot_type;
    const size_t rb_g = ggml_row_size((ggml_type) t_gate, H), rb_u = ggml_row_size((ggml_type) t_up, H);
    const size_t rb_d = ggml_row_size((ggml_type) t_down, F);
    const size_t xq_rb = ggml_row_size(vx, H);
    std::vector<uint8_t> xq(xq_rb * n_rows);
    for (int r = 0; r < n_rows; ++r) ggml_get_type_traits_cpu(vx)->from_float(x + (size_t) r * H, xq.data() + r * xq_rb, H);
    const int n_pairs = u_off[n_u];
    std::vector<float> g((size_t) n_pairs * F), u((size_t) n_pairs * F);
    const int n_gu = n_u * F;
#pragma omp parallel for schedule(static)
    for (int i = 0; i < 2 * n_gu; ++i) {
        const int j = i % n_gu, e = j / F, rr = j % F;
        const bool is_g = i < n_gu;
        const uint8_t* wrow = is_g ? (const uint8_t*) gate[e] + rr * rb_g : (const uint8_t*) up[e] + rr * rb_u;
        for (int p = u_off[e]; p < u_off[e + 1]; ++p) {
            float* dst = is_g ? &g[(size_t) p * F + rr] : &u[(size_t) p * F + rr];
            (is_g ? tg : tu)->vec_dot(H, dst, 0, wrow, 0, xq.data() + pair_row[p] * xq_rb, 0, 1);
        }
    }
    std::vector<float> h((size_t) n_pairs * F);
    for (size_t j = 0; j < h.size(); ++j) h[j] = g[j] / (1.0f + std::exp(-g[j])) * u[j];
    const size_t hq_rb = ggml_row_size(vh, F);
    std::vector<uint8_t> hq(hq_rb * n_pairs);
    for (int p = 0; p < n_pairs; ++p) ggml_get_type_traits_cpu(vh)->from_float(&h[(size_t) p * F], hq.data() + p * hq_rb, F);
    std::memset(y, 0, sizeof(float) * (size_t) n_rows * H);
#pragma omp parallel for schedule(static)
    for (int c = 0; c < H; ++c) {
        for (int e = 0; e < n_u; ++e) {
            const uint8_t* wrow = (const uint8_t*) down[e] + c * rb_d;
            for (int p = u_off[e]; p < u_off[e + 1]; ++p) {
                float s;
                td->vec_dot(F, &s, 0, wrow, 0, hq.data() + p * hq_rb, 0, 1);
                y[(size_t) pair_row[p] * H + c] += pair_w[p] * s;
            }
        }
    }
    return 0;
}

}  // extern "C"

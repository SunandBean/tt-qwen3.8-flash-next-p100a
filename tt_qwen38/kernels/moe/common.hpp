// S2 sparse resident-expert kernels: shared constants.
// ctl (uint32 row-major [1,1,1,1024], written by the host each pass):
//   [0] = count of active resident slots, [1] = rows T,
//   [2 + j] = resident slot index of active entry j (j < count, count <= 256),
//   [CTL_ROWS + r*20 + 2*i], [.. + 1] = (active entry j, fp32 bits of the routing weight) of row r's i-th resident
//   pick (i < 10; j = 0xFFFFFFFF when unused).
#pragma once
#include <cstdint>

constexpr uint32_t CTL_WORDS = 1024;
constexpr uint32_t CTL_SLOTS = 2;
constexpr uint32_t CTL_ROWS = 258;
constexpr uint32_t KT = 80;        // H = 2560
constexpr uint32_t FT = 20;        // F = 640
constexpr uint32_t GU_NT = 40;     // gate_up width 1280
constexpr uint32_t CHUNK = 8;      // weight tiles per CB push

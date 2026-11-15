// S2 fused GDN commit writer: new state tiles (a, b) of head h -> S page h*16 + a*4 + b (in place; this core has read
// all 16 of its S0 tiles before the first one is computed).
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

constexpr uint32_t cb_out = 16;

void kernel_main() {
    constexpr auto s_a = TensorAccessorArgs<0>();
    const uint32_t h = get_arg_val<uint32_t>(0);
    const uint32_t s_addr = get_arg_val<uint32_t>(1);
    const uint32_t tb = get_tile_size(cb_out);
    const auto s_acc = TensorAccessor(s_a, s_addr, tb);
    Noc noc;
    CircularBuffer ocb(cb_out);
    for (uint32_t t = 0; t < 16; t++) {
        ocb.wait_front(1);
        auto src = use<CircularBuffer::AddrSelector::READ_PTR>(ocb);
        noc.async_write(src, s_acc, tb, {.offset_bytes = 0}, {.page_id = h * 16 + t});
        noc.async_write_barrier();
        ocb.pop_front(1);
    }
}

// S2 hyper-connection write writer: out_b tile c -> page b*80 + c of [1,4,32,2560].
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

constexpr uint32_t cb_out = 16;

void kernel_main() {
    constexpr auto o_a = TensorAccessorArgs<0>();
    const uint32_t c = get_arg_val<uint32_t>(0);
    const uint32_t o_addr = get_arg_val<uint32_t>(1);
    const uint32_t tb = get_tile_size(cb_out);
    const auto o_acc = TensorAccessor(o_a, o_addr, tb);
    Noc noc;
    CircularBuffer ocb(cb_out);
    for (uint32_t b = 0; b < 4; b++) {
        ocb.wait_front(1);
        auto src = use<CircularBuffer::AddrSelector::READ_PTR>(ocb);
        noc.async_write(src, o_acc, tb, {.offset_bytes = 0}, {.page_id = b * 80 + c});
        noc.async_write_barrier();
        ocb.pop_front(1);
    }
}

// S2 GDN conv writer: conv tile j and E tile j of this core's channel tiles.
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

constexpr uint32_t cb_conv = 16, cb_Eo = 17;

void kernel_main() {
    constexpr auto c_a = TensorAccessorArgs<0>();
    constexpr auto e_a = TensorAccessorArgs<c_a.next_compile_time_args_offset()>();
    const uint32_t j0 = get_arg_val<uint32_t>(0);
    const uint32_t n = get_arg_val<uint32_t>(1);
    const uint32_t c_addr = get_arg_val<uint32_t>(2), e_addr = get_arg_val<uint32_t>(3);
    const uint32_t tb = get_tile_size(cb_conv);
    const auto c_acc = TensorAccessor(c_a, c_addr, tb);
    const auto e_acc = TensorAccessor(e_a, e_addr, get_tile_size(cb_Eo));
    Noc noc;
    CircularBuffer ccb(cb_conv), ecb(cb_Eo);
    for (uint32_t i = 0; i < n; i++) {
        ecb.wait_front(1);
        noc.async_write(use<CircularBuffer::AddrSelector::READ_PTR>(ecb), e_acc, get_tile_size(cb_Eo),
                        {.offset_bytes = 0}, {.page_id = j0 + i});
        ccb.wait_front(1);
        noc.async_write(use<CircularBuffer::AddrSelector::READ_PTR>(ccb), c_acc, tb, {.offset_bytes = 0},
                        {.page_id = j0 + i});
        noc.async_write_barrier();
        ecb.pop_front(1);
        ccb.pop_front(1);
    }
}

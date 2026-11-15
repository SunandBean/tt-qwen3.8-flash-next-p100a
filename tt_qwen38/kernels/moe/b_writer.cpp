// Program B writer: NQ output tiles nt0.. of y [1,1,32,2560] (fp32).
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

constexpr uint32_t cb_y = 16;

void kernel_main() {
    constexpr uint32_t NQ = get_compile_time_arg_val(0);
    constexpr auto y_a = TensorAccessorArgs<1>();
    const uint32_t nt0 = get_arg_val<uint32_t>(0);
    const uint32_t y_addr = get_arg_val<uint32_t>(1);
    const uint32_t tb_y = get_tile_size(cb_y);
    const auto y_acc = TensorAccessor(y_a, y_addr, tb_y);
    Noc noc;
    CircularBuffer ycb(cb_y);
    ycb.wait_front(NQ);
    auto src = use<CircularBuffer::AddrSelector::READ_PTR>(ycb);
    for (uint32_t q = 0; q < NQ; q++) {
        noc.async_write(src, y_acc, tb_y, {.offset_bytes = q * tb_y}, {.page_id = nt0 + q});
    }
    noc.async_write_barrier();
    ycb.pop_front(NQ);
}

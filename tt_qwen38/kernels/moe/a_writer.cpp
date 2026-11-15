// Program A writer: H tile of work item (j, c) -> Hbuf page j*20 + c ([1, 256, 32, 640] bf16).
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

constexpr uint32_t FT = 20;
constexpr uint32_t cb_ctl = 4, cb_h = 16;

void kernel_main() {
    constexpr auto h_a = TensorAccessorArgs<0>();
    const uint32_t core = get_arg_val<uint32_t>(0);
    const uint32_t ncores = get_arg_val<uint32_t>(1);
    const uint32_t h_addr = get_arg_val<uint32_t>(2);
    const uint32_t tb_h = get_tile_size(cb_h);
    const auto h_acc = TensorAccessor(h_a, h_addr, tb_h);
    Noc noc;
    CircularBuffer ccb(cb_ctl);
    ccb.wait_front(1);
    const uint32_t count = reinterpret_cast<volatile uint32_t*>(ccb.get_read_ptr())[0];
    CircularBuffer hcb(cb_h);
    for (uint32_t i = core; i < count * FT; i += ncores) {
        hcb.wait_front(1);
        auto src = use<CircularBuffer::AddrSelector::READ_PTR>(hcb);
        noc.async_write(src, h_acc, tb_h, {.offset_bytes = 0}, {.page_id = i});
        noc.async_write_barrier();
        hcb.pop_front(1);
    }
}

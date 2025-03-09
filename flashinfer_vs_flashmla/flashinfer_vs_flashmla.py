# code source: https://gist.github.com/abcdabcd987/b215c5f00f4b5e8399b95d7933bcf475
# Results: https://docs.google.com/spreadsheets/d/1t0Txa7Ph9u7Su9LyWpS24vqr9A5FB-FyL0EZNpYOqwg/edit?gid=0#gid=0
# FlashInfer: 28053ac54023fbf3fb552f7be015b0f90a37ed76
# FlashMLA  : accc1695ee0ff996ec63eaf2ebcbf6874ed0e7df
import itertools

import torch
from flash_mla import flash_mla_with_kvcache, get_mla_metadata
from flashinfer import BatchMLAPagedAttentionWrapper
from triton.testing import do_bench  # type: ignore[import]


def cal_diff(x: torch.Tensor, y: torch.Tensor) -> None:
    x, y = x.double(), y.double()
    rmse = ((x - y) * (x - y)).mean().sqrt().item()
    cos_diff = 1 - 2 * (x * y).sum().item() / max((x * x + y * y).sum().item(), 1e-12)
    amax_diff = (x - y).abs().max().item()
    # print(f"{cos_diff=}, {rmse=}, {amax_diff=}")
    assert cos_diff < 1e-5


@torch.inference_mode()
def bench_ragged_vs_mla(
    num_heads: int,
    q_len: int,
    kv_len: int,
    batch_size: int,
) -> None:
    torch.manual_seed(0xABCDABCD987)
    torch.cuda.manual_seed_all(0xABCDABCD987)
    dtype = torch.bfloat16
    device = torch.device("cuda:0")

    page_len = 64
    kv_lora_rank = 512
    qk_nope_head_dim = 128
    qk_rope_head_dim = 64
    # sm_scale = (qk_nope_head_dim + qk_rope_head_dim) ** -0.5
    sm_scale = (kv_lora_rank + qk_rope_head_dim) ** -0.5

    # Inputs for FlashInfer
    num_pages = (kv_len + page_len - 1) // page_len * batch_size
    kv_cache = torch.randn(
        num_pages, page_len, 1, kv_lora_rank, dtype=dtype, device=device
    )
    pe_cache = torch.randn(
        num_pages, page_len, 1, qk_rope_head_dim, dtype=dtype, device=device
    )
    page_indices = torch.arange(num_pages, dtype=torch.int32, device=device)
    page_indptr = torch.tensor(
        [(kv_len + page_len - 1) // page_len * i for i in range(batch_size + 1)],
        dtype=torch.int32,
        device=device,
    )
    kv_len_arr = torch.full((batch_size,), kv_len, dtype=torch.int32, device=device)
    input_indptr = torch.tensor(
        [q_len * i for i in range(batch_size + 1)], dtype=torch.int32, device=device
    )

    fa2 = BatchMLAPagedAttentionWrapper(
        torch.empty(192 << 20, dtype=torch.uint8, device=device), backend="fa2"
    )
    fa3 = BatchMLAPagedAttentionWrapper(
        torch.empty(192 << 20, dtype=torch.uint8, device=device), backend="fa3"
    )
    for mla in [fa2, fa3]:
        mla.plan(
            qo_indptr=input_indptr,
            kv_indptr=page_indptr,
            kv_indices=page_indices,
            kv_len_arr=kv_len_arr,
            num_heads=num_heads,
            head_dim_ckv=kv_lora_rank,
            head_dim_kpe=qk_rope_head_dim,
            page_size=page_len,
            causal=True,
            sm_scale=sm_scale,
            q_data_type=dtype,
            kv_data_type=dtype,
        )

    q_nope = torch.randn(
        q_len * batch_size, num_heads, kv_lora_rank, dtype=dtype, device=device
    )
    q_rope = torch.randn(
        q_len * batch_size, num_heads, qk_rope_head_dim, dtype=dtype, device=device
    )

    # Inputs for FlashMLA
    block_table = page_indices.view(batch_size, (kv_len + page_len - 1) // page_len)
    blocked_k = torch.concat([kv_cache, pe_cache], dim=-1)
    tile_scheduler_metadata, num_splits = get_mla_metadata(
        kv_len_arr, q_len * num_heads, 1
    )
    q = torch.concat([q_nope, q_rope], dim=-1).view(
        batch_size, q_len, num_heads, kv_lora_rank + qk_rope_head_dim
    )

    # Bench

    def run_flashinfer(mla: BatchMLAPagedAttentionWrapper) -> torch.Tensor:
        return mla.run(
            q_nope,
            q_rope,
            kv_cache.squeeze(2),
            pe_cache.squeeze(2),
        ).view(q_len * batch_size, num_heads, kv_lora_rank)

    def run_flashmla() -> torch.Tensor:
        o, lse = flash_mla_with_kvcache(
            q,
            blocked_k,
            block_table,
            kv_len_arr,
            kv_lora_rank,
            tile_scheduler_metadata,
            num_splits,
            causal=True,
        )
        return o.view(q_len * batch_size, num_heads, kv_lora_rank)

    # cal_diff(run_flashinfer(fa2), run_flashmla())
    # cal_diff(run_flashinfer(fa3), run_flashmla())

    l_fa2 = do_bench(lambda: run_flashinfer(fa2)) * 1e3  # type: ignore
    l_fa3 = do_bench(lambda: run_flashinfer(fa3)) * 1e3  # type: ignore
    l_flashmla = do_bench(run_flashmla) * 1e3  # type: ignore

    row = [
        num_heads,
        q_len,
        kv_len,
        batch_size,
        f"{l_fa2:.1f}",
        f"{l_fa3:.1f}",
        f"{l_flashmla:.1f}",
    ]
    print("\t".join(map(str, row)))


def main():
    num_heads_list = [128, 64, 32, 16, 8]
    q_len_list = [1]
    batch_size_list = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    kv_len_list = [128, 1024, 4096, 8192, 16384, 32768, 65536, 65536*2]

    header = ["nhead", "q_len", "kv_len", "bs", "FA2", "FA3", "FlashMLA"]
    print("\t".join(header))
    for num_heads, q_len, kv_len, batch_size in itertools.product(
        num_heads_list, q_len_list, kv_len_list, batch_size_list
    ):
        bench_ragged_vs_mla(num_heads, q_len, kv_len, batch_size)


if __name__ == "__main__":
    main()
运行脚本：

- 跑benchmark

`python3 -m mla.benchmark B 1024`

`python3 -m mla.benchmark B 1024 --repeat=1`

- 跑性能
`python workspace/blog/data/runner.py`

- 跑一致性
`python3 -m mla.test B 1024 --repeat=1`

跑 flashinfer 前需要 `export TORCH_CUDA_ARCH_LIST="9.0+PTX"`


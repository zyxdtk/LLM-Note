# 1. 大模型推理学习资料

- [LLM推理优化技术详解](https://zhuanlan.zhihu.com/p/655557420) 推理加速方法
- [github:vLLM](https://github.com/vllm-project/vllm) 高性能推理框架
- [NVIDIA/FasterTransformer](https://github.com/NVIDIA/FasterTransformer)
- [分析transformer模型的参数量、计算量、中间激活、KV cache](https://zhuanlan.zhihu.com/p/624740065)

# 大模型部署框架

- [54.7k] [vllm](https://github.com/vllm-project/vllm)
- [16.8k] [sglang](https://github.com/sgl-project/sglang)
- [150k] [ollama](https://github.com/ollama/ollama)
- [84.5] [llama.cpp](https://github.com/ggml-org/llama.cpp)
- [21.9] [mlx](https://github.com/ml-explore/mlx)
- [14.8k] [ktransformers](https://github.com/kvcache-ai/ktransformers)
- [0] [lmstudio](https://lmstudio.ai/)


# 2. 大模型推理优化


大模型推理关注：延迟、吞吐和成本，优化分：改模型参数、单机优化、分布式优化。

- 改模型参数。
    - 量化
    - attention结构(mha、mqa、mla、sparse attention、 liner attention)
    - ffn结构(moe)
    - 其他结构(silu、rmsnorm)
    - 随机解码。
- 单机优化。LLM是io约束的。
    - 算子融合。qkv融合，bias融合。
    - 高性能算子。flash attention、高性能矩阵运算gemm。需要深入到kernel层面。
    - 内存管理。continuous batching、paged attention。
- 分布式优化。
    - 模型并行。tensor并行、pipeline并行、专家并行
    - 数据并行。zero3
    - 硬件特化。prefill和generate分离。

## 2.1. 改模型参数

### 2.1.1. 量化

#### 论文
- [2025.01] [Qrazor: Reliable and Effortless 4-bit LLM Quantization by Significant Data Razoring](https://arxiv.org/abs/2501.13331)
- [2024.08] [MARLIN: Mixed-Precision Auto-Regressive Parallel Inference on Large Language Models](https://arxiv.org/abs/2408.11743)
- [2023.10] [LLM-FP4: 4-Bit Floating-Point Quantized Transformers](https://arxiv.org/abs/2310.16836)
- [2023.09] [Optimize Weight Rounding via Signed Gradient Descent for the Quantization of LLMs](https://arxiv.org/abs/2309.05516) AutoRound
    - [intel/auto-round](https://github.com/intel/auto-round)
- [2023.06] [AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration](https://arxiv.org/abs/2306.00978) AWQ
- [2022.10] [GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers](https://arxiv.org/abs/2210.17323) GPTQ
- [2022.08] [LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale](https://arxiv.org/abs/2208.07339)
    - [LLM.int8(): 8-bit量化推理](https://arxiv.org/abs/2208.07339)
    - [LLM大模型之精度问题（FP16，FP32，BF16）详解与实践](https://zhuanlan.zhihu.com/p/657886517)
#### 工具
- [1.8k] [vllm-project/llm-compressor](https://github.com/vllm-project/llm-compressor/)
- [730] [ModelCloud/GPTQModel](https://github.com/ModelCloud/GPTQModel)


### 2.1.2. attention结构

- [FlashMLA](https://github.com/deepseek-ai/FlashMLA) 

### 2.1.3. 并行解码

- [2025.03] [EAGLE-3: Scaling up Inference Acceleration of Large Language Models via Training-Time Test](https://arxiv.org/abs/2503.01840) 5.6倍加速
    - [hemingkx/SpeculativeDecodingPapers](https://github.com/hemingkx/SpeculativeDecodingPapers)
    - [SafeAILab/EAGLE](https://github.com/SafeAILab/EAGLE) eagle-3相比原始有5.6倍加速
    - [sgl-project/SpecForge](https://github.com/sgl-project/SpecForge)
    - [angelslim-benchmark](https://angelslim.readthedocs.io/zh-cn/latest/performance/speculative_decoding/benchmarks.html)
    - [eagle3部署：A100-cuda12.6-sglang0.4.6](https://zhuanlan.zhihu.com/p/1938989829766509156)
- [2024.01] [Unlocking Efficiency in Large Language Model Inference:A Comprehensive Survey of Speculative Decoding](https://arxiv.org/abs/2401.07851) 综述
    - [万字综述 10+ 种 LLM 投机采样推理加速方案](https://www.53ai.com/news/finetuning/2024071109285.html)
- [2024.01] [EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty](https://arxiv.org/abs/2401.15077) 预估特征层
    - [论文解读】EAGLE：在特征层进行自回归的投机采样框架](https://zhuanlan.zhihu.com/p/15955544919)
    - [vllm-spec_decode](https://docs.vllm.ai/en/v0.9.0/features/spec_decode.html)
- [2024.01] [Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads](https://arxiv.org/abs/2401.10774) 加多个解码头，用topk解码多个token，用tree attention判定是否采纳。
- [2023.10] [Lookahead: An Inference Acceleration Framework for Large Language Model with Lossless Generation Accuracy](https://arxiv.org/abs/2312.12728) 用2D窗口维护多个ngram
- [2022.11] [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192) 小模型预估，大模型判定是否采纳。计算量不变，但是可以并行化了。


## 2.2. 单机优化

### 2.2.1. attention

- [2025.01] [FlashInfer: Efficient and Customizable Attention Engine for LLM Inference Serving](https://arxiv.org/abs/2501.01005)
    - [flashinfer](https://github.com/flashinfer-ai/flashinfer)
- [2023.09] [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180) PagedAttention,虚拟内存技术，分页。比朴素batch快22倍吞吐，比ft快4倍。
    - [vllm-project/vllm](https://github.com/vllm-project/vllm)
    - [How continuous batching enables 23x throughput in LLM inference while reducing p50 latency](https://www.anyscale.com/blog/continuous-batching-llm-inference)
- [2023.08] [SARATHI: Efficient LLM Inference by Piggybacking Decodes with Chunked Prefills
](https://arxiv.org/abs/2308.16369) Chunk Prefills 
    - [LLM推理优化 - Chunked prefills](https://zhuanlan.zhihu.com/p/14689463165)
- [2022.05] [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135) FlashAttention，提速2.4倍
 

### 2.2.2. FFN


- [2023.06] [MoE: An Efficient Mixture of Experts for Large Language Models](URL_ADDRESS- [2023.06] [MoE: An Efficient Mixture of Experts for Large Language Models](https://arxiv.org/abs/2306.05087)
- [GEMM](https://github.com/iVishalr/GEMM) 矩阵算子优化
- [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM) FP8矩阵算子优化
- [DeepEP](https://github.com/deepseek-ai/DeepEP) 专家并行


## 2.3. 分布式优化

- [3FS](https://github.com/deepseek-ai/3FS) 分布式文件系统
- [2022.12] [Overlap Communication with Dependent Computation via Decomposition in Large Deep Learning Models](https://dl.acm.org/doi/abs/10.1145/3567955.3567959)
- [2022.11] [Efficiently Scaling Transformer Inference](https://arxiv.org/abs/2211.05102) 
    - [大模型并行推理的太祖长拳：解读Jeff Dean署名MLSys 23杰出论文](https://zhuanlan.zhihu.com/p/660715870)
- [2022.07] [Alpa: Automating Inter- and Intra-Operator Parallelism for Distributed Deep Learning](https://www.usenix.org/system/files/osdi22-zheng-lianmin.pdf)
- [大模型推理序列并行](https://zhuanlan.zhihu.com/p/9816504195)
- [序列并行DeepSpeed-FPDT](https://zhuanlan.zhihu.com/p/720387198)
- [Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving](https://arxiv.org/abs/2407.00079) PD分离
    - [kvcache-ai/Mooncake](https://github.com/kvcache-ai/Mooncake)
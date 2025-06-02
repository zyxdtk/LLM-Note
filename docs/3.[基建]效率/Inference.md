# 1. 大模型推理学习资料

- [LLM推理优化技术详解](https://zhuanlan.zhihu.com/p/655557420) 推理加速方法
- [github:vLLM](https://github.com/vllm-project/vllm) 高性能推理框架
- [分析transformer模型的参数量、计算量、中间激活、KV cache](https://zhuanlan.zhihu.com/p/624740065)

# 2. 大模型推理优化

大模型推理优化有一个基础认知，LLM是一个io约束。
大模型推理优化分：改模型参数、单机优化、分布式优化、调度优化。

- 改模型参数。
    - 量化
    - attention结构(mha、mqa、mla、sparse attention、 liner attention)
    - ffn结构(moe)
    - 其他结构(silu、rmsnorm)
    - 随机解码。
- 单机优化。
    - 算子融合。qkv融合，bias融合。
    - 高性能算子。flash attention、高性能矩阵运算gemm。需要深入到kernel层面。
    - 调度优化。continuous batching
- 分布式优化。
    - 模型并行。tensor并行、pipeline并行、专家并行
    - 数据并行。zero3
    - 硬件特化。prefill和generate分离。

## 2.1. 改模型参数

### 2.1.1. 量化

- [2025.01] [Qrazor: Reliable and Effortless 4-bit LLM Quantization by Significant Data Razoring](https://arxiv.org/abs/2501.13331)
- [2023.10] [LLM-FP4: 4-Bit Floating-Point Quantized Transformers](https://arxiv.org/abs/2310.16836)
- [2023.09] [Optimize Weight Rounding via Signed Gradient Descent for the Quantization of LLMs](https://arxiv.org/abs/2309.05516) AutoRound
    - [intel/auto-round](https://github.com/intel/auto-round)
- [2023.06] [AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration](https://arxiv.org/abs/2306.00978) AWQ
- [2022.10] [GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers](https://arxiv.org/abs/2210.17323) GPTQ
- [2022.08] [LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale](https://arxiv.org/abs/2208.07339)
    - [LLM.int8(): 8-bit量化推理](https://arxiv.org/abs/2208.07339)

### 2.1.2. attention结构

- [FlashMLA](https://github.com/deepseek-ai/FlashMLA) 

### 2.1.3. 并行解码


- [2024.01] [EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty](https://arxiv.org/abs/2401.15077)
    - [论文解读】EAGLE：在特征层进行自回归的投机采样框架](https://zhuanlan.zhihu.com/p/15955544919)
- [2024.01] [Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads](https://arxiv.org/abs/2401.10774) 加多个解码头，用topk解码多个token，用tree attention判定是否采纳。
- [2023.10] [Lookahead: An Inference Acceleration Framework for Large Language Model with Lossless Generation Accuracy](https://arxiv.org/abs/2312.12728) 用2D窗口维护多个ngram
- [2022.11] [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192) 小模型预估，大模型判定是否采纳。计算量不变，但是可以并行化了。


## 2.2. 单机优化

### 2.2.1. attention

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
- [2022.11] [Efficiently Scaling Transformer Inference](https://arxiv.org/abs/2211.05102) 
- [大模型推理序列并行](https://zhuanlan.zhihu.com/p/9816504195)
- [序列并行DeepSpeed-FPDT](https://zhuanlan.zhihu.com/p/720387198)

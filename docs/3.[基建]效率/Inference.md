## 1. 大模型推理学习资料

- [LLM推理优化技术详解](https://example.com/llm-inference) 推理加速方法
- [github:vLLM](https://github.com/vllm-project/vllm) 高性能推理框架

## 2. 大模型推理优化

### 2.1. 分布式优化

- [3FS](https://github.com/deepseek-ai/3FS) 分布式文件系统
- [2022.11] [Efficiently Scaling Transformer Inference](https://arxiv.org/abs/2211.05102) 

### 2.2. 服务器并发优化

- [How continuous batching enables 23x throughput in LLM inference while reducing p50 latency](https://www.anyscale.com/blog/continuous-batching-llm-inference)

### 2.3. 显存优化

- [FlashAttention: 高效注意力机制](https://arxiv.org/abs/2205.14135)

### 2.4. 算子优化

- [GEMM](https://github.com/iVishalr/GEMM) 矩阵算子优化
- [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM) FP8矩阵算子优化
- [FlashMLA](https://github.com/deepseek-ai/FlashMLA) 
- [DeepEP](https://github.com/deepseek-ai/DeepEP) 专家并行

### 2.5. 量化

- [LLM学习1：大模型架构要点总结](https://zhuanlan.zhihu.com/p/20329244481)
- [LLM.int8(): 8-bit量化推理](https://arxiv.org/abs/2208.07339)

### 2.6. 并行解码


- [2024.01] [EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty](https://arxiv.org/abs/2401.15077)
    - [论文解读】EAGLE：在特征层进行自回归的投机采样框架](https://zhuanlan.zhihu.com/p/15955544919)
- [2024.01] [Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads](https://arxiv.org/abs/2401.10774) 加多个解码头，用topk解码多个token，用tree attention判定是否采纳。
- [2023.10] [Lookahead: An Inference Acceleration Framework for Large Language Model with Lossless Generation Accuracy](https://arxiv.org/abs/2312.12728) 用2D窗口维护多个ngram
- [2022.11] [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192) 小模型预估，大模型判定是否采纳。计算量不变，但是可以并行化了。


## 3. 大模型推理服务部署

- [分析transformer模型的参数量、计算量、中间激活、KV cache](https://zhuanlan.zhihu.com/p/624740065)
## 大模型学习资料

### 入门教程

- [LLM推理优化技术详解](https://example.com/llm-inference) 推理加速方法
- [github:vLLM](https://github.com/vllm-project/vllm) 高性能推理框架

### 经典论文
- [FlashAttention: 高效注意力机制](https://arxiv.org/abs/2205.14135)
- [LLM.int8(): 8-bit量化推理](https://arxiv.org/abs/2208.07339)

## 大模型推理核心技术

### 1. 推理优化技术
- 量化压缩（INT8/FP8/INT4）
    - [LLM学习1：大模型架构要点总结](https://zhuanlan.zhihu.com/p/20329244481)
- KV Cache缓存机制
- 注意力优化（FlashAttention等）

### 2. 解码策略
- 贪心搜索 vs 束搜索
- 采样技术（Top-k, Top-p）
- 温度调节策略

### 3. 服务部署
- API服务化（FastAPI/GRPC）
- 批处理优化技术
- 硬件加速方案

### 4. 性能评估
- 延迟与吞吐量指标
- 显存占用分析
- 计算利用率监控

## 主流推理框架
| 框架名称 | 特点 | 适用场景 |
|---------|------|---------|
| vLLM    | PagedAttention优化 | 高并发服务 |
| TGI      | HuggingFace官方 | 快速部署 |
| TensorRT-LLM | NVIDIA优化 | 专业硬件加速 |
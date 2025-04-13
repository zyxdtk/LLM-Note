
## 大模型学习资料

### 入门教程

- [LLM学习1：大模型架构要点总结](https://zhuanlan.zhihu.com/p/648050614) 回忆基础知识
- [github:llm-viz](https://github.com/bbycroft/llm-viz)/[网页:bbycroft](https://bbycroft.net/llm)大模型结构可视化

### 经典论文

- [2025.03] [Gemma 3 Technical Report](https://storage.googleapis.com/deepmind-media/gemma/Gemma3Report.pdf) 多模态理解、蒸馏和量化
- [2024.12] [DeepSeek-V3 Technical Report](https://arxiv.org/pdf/2412.19437)/[github:DeepSeek-V3](https://github.com/deepseek-ai/DeepSeek-V3) MLA、MOE、MTP、GRPO等



## 大模型预训练核心模块

### 1. 数据预处理
- 数据清洗与去重
- 文本标准化（大小写、标点等）
- Tokenization
    - byte-pair-encoding (BPE)  
        - [2016] [Neural Machine Translation of Rare Words with Subword Units](https://aclanthology.org/P16-1162/) bpe用subword来处理oov问题。把词打散成char，词尾需要添加特殊字符<\w>。 通过合并最频繁出现的相邻子词对来迭代地构建更大的子词单元。
        - [github:subword-nmt](https://github.com/rsennrich/subword-nmt)
    - WordPiece
        - [2016.10] [Google’s Neural Machine Translation System: Bridging the Gap
between Human and Machine Translation](https://arxiv.org/pdf/1609.08144) 基于lstm的8层encoder-decoder模型处理翻译任务,用到了残差。提出了wordpiece, 在词首添加_词首符号。通过概率最大化选择子词对。
    - SentencePiece
        - [github:google/sentencepiece](https://github.com/google/sentencepiece)  NFKC-based normalization, 
        - [2018.08] [SentencePiece: A simple and language independent subword tokenizer and detokenizer for Neural Text Processing](https://arxiv.org/abs/1808.06226) 
        - [2018.04] [Subword Regularization: Improving Neural Network Translation Models with Multiple Subword Candidates](https://arxiv.org/abs/1804.10959) 每次训练时从概率分布中随机采样一种分割方式作为输入，而非固定使用最高概率分割。
        - [2019.10] [BPE-Dropout: Simple and Effective Subword Regularization](https://arxiv.org/abs/1910.13267)  以概率 p 随机跳过某些合并步骤。
- 数据分布分析

### 2. 模型架构
- Transformer结构选择（Encoder/Decoder/Encoder-Decoder）
- 注意力机制变体（多头注意力、稀疏注意力等）
- 位置编码方案（绝对/相对位置编码）
    - rope
        - [2021.03] [Transformer升级之路：2、博采众长的旋转式位置编码](https://kexue.fm/archives/8265) 每2位做一个旋转，旋转角度为1/2^k, 其中k为位置。qk相乘之后相对距离越远，qk的乘积越小。
        - [旋转矩阵及左右乘的意义，看这一篇就够了](https://blog.csdn.net/weixin_45632220/article/details/117735223) 
- 归一化层选择（LayerNorm/RMSNorm）


### 3. 训练策略
- 优化器选择（Adam/AdamW/LAMB）
- 学习率调度（线性预热+余弦衰减）
- 批次策略（动态批处理/梯度累积）
- 混合精度训练（FP16/BF16）

### 4. 分布式训练
- 数据并行（Data Parallelism）
- 流水线并行（Pipeline Parallelism）
- 张量并行（Tensor Parallelism）
- 3D并行策略组合

### 5. 损失函数
- 语言建模损失（标准交叉熵）
- 掩码语言建模（MLM）
- 序列到序列损失
- 特殊token处理策略

### 6. 监控与调试
- 训练动态监控（损失/梯度/激活值）
- 显存使用分析
- 异常检测（梯度爆炸/消失）
- 模型检查点管理

### 7. 扩展技术
- 课程学习（Curriculum Learning）
- 模型增长（渐进式训练）
- 知识蒸馏（Teacher-Student）
- 持续预训练









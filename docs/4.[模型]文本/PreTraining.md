
## 1. 大模型学习资料

- [LLM学习1：大模型架构要点总结](https://zhuanlan.zhihu.com/p/648050614) 回忆基础知识
- [github:llm-viz](https://github.com/bbycroft/llm-viz)/[网页:bbycroft](https://bbycroft.net/llm)大模型结构可视化


## 2. 论文和开源库

### 2.1. DeepSeek

- [2024.12] [DeepSeek-V3 Technical Report](https://arxiv.org/pdf/2412.19437)/[github:DeepSeek-V3](https://github.com/deepseek-ai/DeepSeek-V3) MLA、MOE、MTP、GRPO等
- [2024.05] [DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model](https://arxiv.org/abs/2405.04434)
    - [工业界主流大语言模型后训练(Post-Training)技术总结](https://zhuanlan.zhihu.com/p/987052830)
- [2024.01] [DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models](https://arxiv.org/pdf/2401.06066)
- [2024.01] [DeepSeek LLM: Scaling Open-Source Language Models with Longtermism](https://arxiv.org/abs/2401.02954) 

### 2.2. Google

- [2025.03] [Gemma 3 Technical Report](https://storage.googleapis.com/deepmind-media/gemma/Gemma3Report.pdf) 多模态理解、蒸馏和量化

### 2.3. Openai

- [2024.03] [GPT-4 Technical Report](https://arxiv.org/pdf/2303.08774) 

### 2.4. 智谱AI

- [2024.07] [ChatGLM: A Family of Large Language Models from GLM-130B to GLM-4 All Tools](https://arxiv.org/pdf/2406.12793)
    - [GLM-4](https://github.com/THUDM/GLM-4) 



## 3. 大模型预训练核心模块

### 3.1. 数据预处理
- 数据清洗
    - [kenlm](https://kheafield.com/code/kenlm/) 速度快，占用内存小，支持多线程。用优质语料训练模型，然后用来过滤低质量语料。
        - [自然语言处理之数据平滑方法](https://blog.csdn.net/fuermolei/article/details/81353746) 第一种类型为政府给大家每人一笔或者几笔钱（如1和2），第二种为找父母要（如3和4），最后一种就是劫富济贫（如5-7）。比喻很好，最后kn平滑的公式图不对。
        - [Kenlm中使用的Modified Kneser-Ney 平滑方法和计算过程推演](https://zhuanlan.zhihu.com/p/406029473) 把kn的公式推演了一遍，跟上面文章的结合看会比较好理解。
        - [github:kenlm](https://github.com/kpu/kenlm) kenlm的c++实现,官方库
        - [github:kneser-ney](https://github.com/smilli/kneser-ney) kn的python实现
        - [Scalable Modified Kneser-Ney Language Model Estimation](https://kheafield.com/papers/edinburgh/estimate_paper.pdf) 比srilm用7.7%的ram和14%的时间。介绍了kn的优化。
        - [KenLM: Faster and Smaller Language Model Queries](https://kheafield.com/papers/avenue/kenlm.pdf) 介绍了kenlm的优化。trie数存储n-gram概率降序排列，bit-level压缩存储概率和backoff，变长编码存储n-gram索引。与计算边界条件概率、用sse指令并行计算、延迟backoff计算。mmio实现零拷贝加载。
- 数据去重 
- 文本标准化（大小写、标点等）
- 数据增强
    - [2024.04] [Fewer Truncations Improve Language Modeling](https://arxiv.org/abs/2404.10830)
    - [2022.07] [Efficient Training of Language Models to Fill in the Middle](https://arxiv.org/abs/2207.14255)

### 3.2. Tokenization

介绍和代码库：

- [大模型基础组件 - Tokenizer](https://zhuanlan.zhihu.com/p/651430181)  详细介绍了bpe、bbpe、wordpiece、sentencepiece
- [huggingface/tokenizers](https://github.com/huggingface/tokenizers)

算法：

- byte-pair-encoding (BPE)  
    - [2016] [Neural Machine Translation of Rare Words with Subword Units](https://aclanthology.org/P16-1162/) bpe用subword来处理oov问题。把词打散成char，词尾需要添加特殊字符<\w>。 通过合并最频繁出现的相邻子词对来迭代地构建更大的子词单元。
    - [github:subword-nmt](https://github.com/rsennrich/subword-nmt)
- WordPiece
    - [2016.10] [Google’s Neural Machine Translation System: Bridging the Gap
between Human and Machine Translation](https://arxiv.org/pdf/1609.08144) 基于lstm的8层encoder-decoder模型处理翻译任务,用到了残差。提出了wordpiece, 在词首添加_词首符号。通过概率最大化选择子词对。
- BBPE
    - [2019.09] [Neural Machine Translation with Byte-Level Subwords](https://arxiv.org/abs/1909.03341)
- SentencePiece
    - [github:google/sentencepiece](https://github.com/google/sentencepiece)  NFKC-based normalization, 
    - [2018.08] [SentencePiece: A simple and language independent subword tokenizer and detokenizer for Neural Text Processing](https://arxiv.org/abs/1808.06226) 
    - [2018.04] [Subword Regularization: Improving Neural Network Translation Models with Multiple Subword Candidates](https://arxiv.org/abs/1804.10959) 每次训练时从概率分布中随机采样一种分割方式作为输入，而非固定使用最高概率分割。
    - [2019.10] [BPE-Dropout: Simple and Effective Subword Regularization](https://arxiv.org/abs/1910.13267)  以概率 p 随机跳过某些合并步骤。


### 3.3. 模型架构
- Transformer结构选择（Encoder/Decoder/Encoder-Decoder）
- 位置编码方案（绝对/相对位置编码）
    - rope
        - [2021.03] [Transformer升级之路：2、博采众长的旋转式位置编码](https://kexue.fm/archives/8265) 每2位做一个旋转，旋转角度为1/2^k, 其中k为位置。qk相乘之后相对距离越远，qk的乘积越小。
        - [旋转矩阵及左右乘的意义，看这一篇就够了](https://blog.csdn.net/weixin_45632220/article/details/117735223) 
- 归一化层选择（LayerNorm/RMSNorm）
    - [LayerNorm VS BatchNorm VS RMSNorm](https://zhuanlan.zhihu.com/p/694909672)
    - [Group Normalization](https://arxiv.org/pdf/1803.08494) 这里是图像中的norm，跟nlp中的还不太一样
    - [2019.10] [Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467) RMSNorm性能和LayerNorm相当，但是可以节省7%到64%的运算。
- 激活函数
    - [激活函数 Relu,Gelu,Mish,SiLU,Swish,Tanh,Sigmoid](https://blog.csdn.net/weixin_38649779/article/details/127647257) deepseek使用silu，SiLU具备无上界有下界、平滑、非单调的特性。SiLU在深层模型上的效果优于 ReLU。可以看做是平滑的ReLU激活函数。
- 长上下文
    - [2023.08] [YaRN: Efficient Context Window Extension of Large Language Models](https://arxiv.org/abs/2309.00071)
        - [论文YaRN: Efficient Context Window Extension of Large Language Models笔记](https://zhuanlan.zhihu.com/p/683863159)


### 激活函数


- [2020.02] [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202) SwiGLU。论文提到预训练不用dropout效果更好。GLU效果更好，作者解释不了，所以归因于上天眷顾
- [2017.10] [Searching for Activation Functions](https://arxiv.org/abs/1710.05941)
- [2016.06] [Gaussian Error Linear Units (GELUs)](https://arxiv.org/abs/1606.08415)


### 3.4. 注意力机制


- [缓存与效果的极限拉扯：从MHA、MQA、GQA到MLA](https://spaces.ac.cn/archives/10091) 节约kv cache空间。
- [2024.01] [Lightning Attention-2: A Free Lunch for Handling Unlimited Sequence Lengths in Large Language Models](https://arxiv.org/abs/2401.04658)
    - [OpenNLPLab/lightning-attention](https://github.com/OpenNLPLab/lightning-attention)
    - [新一代注意力机制Lightning Attention-2：无限序列长度、恒定算力开销、更高建模精度](https://zhuanlan.zhihu.com/p/678552539)
- [2023.07] [TransNormerLLM: A Faster and Better Large Language Model with Improved TransNormer](https://arxiv.org/abs/2307.14995)
- [2020.01] [Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention](https://arxiv.org/abs/2006.16236) 线性注意力
    - [线性Attention的探索：Attention必须有个Softmax吗？](https://spaces.ac.cn/archives/7546)
    - [笔记：简单图解一下线性注意力机制](https://zhuanlan.zhihu.com/p/718156896) SSM(State Space Model) 实现了对每一个历史步骤的记录和压缩，但是忽略了具体的步数索引。
- [2019.04] [Generating Long Sequences with Sparse Transformers](https://arxiv.org/abs/1904.10509) 稀疏注意力
    - [openai/sparse_attention](https://github.com/openai/sparse_attention)
    - [为节约而生：从标准Attention到稀疏Attention](https://spaces.ac.cn/archives/6853) 
    - [Transformer综述（一）：稀疏注意力](https://zhuanlan.zhihu.com/p/691296437)
    - [Sliding Window Attention（滑动窗口注意力）](https://blog.csdn.net/shizheng_Li/article/details/145809397)
    - [2023.09] [Mistral 7B](https://mistral.ai/news/announcing-mistral-7b)

### 3.5. 训练策略
- 优化器选择（Adam/AdamW/LAMB）
- 学习率调度（线性预热+余弦衰减）
- 批次策略（动态批处理/梯度累积）
- 混合精度训练（FP16/BF16）

### 3.6. 分布式训练

- [huggingface: Model Parallelism](https://huggingface.co/docs/transformers/v4.15.0/en/parallelism)
- 数据并行（Data Parallelism）
- 流水线并行（Pipeline Parallelism）
- 张量并行（Tensor Parallelism）
- 3D并行策略组合

### 3.7. 损失函数
- 语言建模损失（标准交叉熵）
- 掩码语言建模（MLM）
- 序列到序列损失
- 特殊token处理策略

### 3.8. 监控与调试
- 训练动态监控（损失/梯度/激活值）
- 显存使用分析
- 异常检测（梯度爆炸/消失）
- 模型检查点管理

### 3.9. 扩展技术
- 课程学习（Curriculum Learning）
- 模型增长（渐进式训练）
- 知识蒸馏（Teacher-Student）
- 持续预训练









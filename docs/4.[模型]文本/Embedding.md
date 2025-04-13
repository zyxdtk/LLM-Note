# 文本Embedding技术

## 学习资料

- [sbert](https://www.sbert.net/) 
    - [用1B个对比对训练最好的句向量模型](https://discuss.huggingface.co/t/train-the-best-sentence-embedding-model-ever-with-1b-training-pairs/7354) 源于这样的认知：更多更多样的数据+更大BatchSize可以训练出更好的Embedding模型。
    - [sentence-transformers](https://github.com/UKPLab/sentence-transformers)

## benchmark

- [MTEB评估基准](https://github.com/embeddings-benchmark/mteb)
    - [mteb/leaderboard](https://huggingface.co/spaces/mteb/leaderboard)

## 论文

### 基于Transformer架构

- [2025.03] [Gemini Embedding: Generalizable Embeddings from Gemini](https://www.arxiv.org/abs/2503.07891) google sota效果。难负例、合成数据蒸馏大模型、用LLM初始化。引入task prompts and a pre-finetuning stage进一步提高效果。用Model Soup来融合多个checkpoint的效果。mean polling 然后过一个linear到目标维度。nce。
- [2024.09] [C-Pack: Packed Resources For General Chinese Embeddings](https://arxiv.org/pdf/2309.07597) 智源开源。分三个阶段：预训练、对比对学习、任务精调。预训练使用retroMAE，用in-batch负采样，batchsize 19200，任务精调用ann挑选难负样本。
    - [FlagEmbedding](https://github.com/FlagOpen/FlagEmbedding)
    - [bge-large-zh-v1.5](https://huggingface.co/BAAI/bge-large-zh-v1.5)
    - [bge-reranker-large](https://huggingface.co/BAAI/bge-reranker-large)
- [2023.08] [Towards General Text Embeddings with Multi-stage Contrastive Learning](https://arxiv.org/abs/2308.03281) 阿里开源。infoNCE。mean polling，两阶段：大batchsize的in-batch训练。难负样本训练。
   - [gte-large-zh](https://huggingface.co/thenlper/gte-large-zh)
- [2022.08] [https://arxiv.org/pdf/2202.08904v5](SGPT: GPT Sentence Embeddings for Semantic Search) 清华，对比了 cross-encoder vs bi-encoder，使用位置加权的mean polling，只是fine-tunning了bias
- [2022.01] [Text and Code Embeddings by Contrastive Pre-Training](https://arxiv.org/pdf/2201.10005) openai 1. 预训练模型作为初始化模型 2. 大batch的对比学习logit，相似度+交叉熵+in-batch负采样 3. Fine-tuning (非必须)
- [2021.04] [SimCSE: Simple Contrastive Learning of Sentence Embeddings](https://arxiv.org/abs/2104.08821)
    - [github:SimCSE](https://github.com/princeton-nlp/SimCSE#model-list)
- [2020.09] [Dense Passage Retrieval for Open-Domain Question Answering](https://arxiv.org/pdf/2004.04906) meta
    - https://huggingface.co/docs/transformers/model_doc/dpr#overview
    - [github:DPR](https://github.com/facebookresearch/DPR)


## 训练技术
### 1. 目标函数
- 对比损失(Contrastive Loss)
- 三重损失(Triplet Loss)
- 余弦相似度损失

### 2. 数据增强
- 回译(Back Translation)
- 词语删除/替换
- 对抗样本生成
- 难例挖掘(Hard Negative Mining)
- 合成数据
    - [2024.5] [Improving Text Embeddings with Large Language Models](https://arxiv.org/abs/2401.00368) 用LLM合成93种语言的finetune数据。
    - [2024.03] [Gecko: Versatile Text Embeddings Distilled from Large Language Models](https://arxiv.org/abs/2403.20327) 从LLM蒸馏知识。用LLM生成QA对，然后对每一个query精挑A，标注正例和难负例。
    - [2022.09] [Promptagator: Few-shot Dense Retrieval From 8 Examples](https://arxiv.org/abs/2209.11755) 用few-shot的prompt来生成query

### 3. 训练技巧

- 层归一化策略
- 温度系数调节
- 输出维度
    - [2022.04] [Matryoshka Representation Learning](https://arxiv.org/abs/2205.13147) 用一个Embedding上构建不同的loss，每个loss使用不同数量和组合的维度。

## 评估指标
| 指标名称       | 说明                  | 典型数据集       |
|----------------|-----------------------|------------------|
| Spearman相关性 | 排名相关性            | STS-B            |
| Recall@K       | 检索召回率            | MS-MARCO         |
| 聚类纯度       | 聚类效果评估          | AG News          |



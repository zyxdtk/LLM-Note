

## 理论

- [2026.01] [From Entropy to Epiplexity: Rethinking Information for Computationally Bounded Intelligence](https://arxiv.org/abs/2601.03220)

## 数据清洗

- [2025.04] [Meta-rater: A Multi-dimensional Data Selection Method for Pre-training Language Models](https://arxiv.org/abs/2504.14194) 四维质量评估（PRRC）; Meta-rater 方法训练多个代理小模型从多个维度打分，最后选出综合质量更高的数据。
- [2024.02] [Clustering and Ranking: Diversity-preserved Instruction Selection through Expert-aligned Quality Estimation](https://arxiv.org/abs/2402.18191) 用rm对qa对打分然后排序。pca降维，kmeans聚类。
- [2023.12] [What Makes Good Data for Alignment? A Comprehensive Study of Automatic Data Selection in Instruction Tuning](https://arxiv.org/abs/2312.15685) deita。complexity, quality, and diversity。用gpt来给指令和QA对打复杂度和质量分，用emb_sim来评估相似度。
    - [论文解读：如何自动选择SFT数据](https://zhuanlan.zhihu.com/p/690779419)
    - [LLM模型之高质量数据选择和微调方法](https://zhuanlan.zhihu.com/p/703612817)
- [2023.08] [InsTag: Instruction Tagging for Analyzing Supervised Fine-tuning of Large Language Models](https://arxiv.org/abs/2308.07074) instag


## 主动学习

- [2024.05] [ActiveLLM: Large Language Model-based Active Learning for Textual Few-Shot Scenarios](https://arxiv.org/abs/2405.10808)




- [ALiPy: Active Learning in Python](https://github.com/NUAA-AL/ALiPy)
    - [https://parnec.nuaa.edu.cn/huangsj/alipy/](https://parnec.nuaa.edu.cn/huangsj/alipy/)
- [baifanxxx/awesome-active-learning](https://github.com/baifanxxx/awesome-active-learning)
- [modAL-python/modAL](https://github.com/modAL-python/modAL)
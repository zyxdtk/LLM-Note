

## 数据清洗

- [2024.02] [Clustering and Ranking: Diversity-preserved Instruction Selection through Expert-aligned Quality Estimation](https://arxiv.org/abs/2402.18191) 用rm对qa对打分然后排序。pca降维，kmeans聚类。
- [2023.12] [What Makes Good Data for Alignment? A Comprehensive Study of Automatic Data Selection in Instruction Tuning](https://arxiv.org/abs/2312.15685) deita。complexity, quality, and diversity。用gpt来给指令和QA对打复杂度和质量分，用emb_sim来评估相似度。
    - [论文解读：如何自动选择SFT数据](https://zhuanlan.zhihu.com/p/690779419)
    - [LLM模型之高质量数据选择和微调方法](https://zhuanlan.zhihu.com/p/703612817)
- [2023.08] [InsTag: Instruction Tagging for Analyzing Supervised Fine-tuning of Large Language Models](https://arxiv.org/abs/2308.07074) instag
# 多模态大模型评测


## 幻觉

- [2025.06] [EditInspector: A Benchmark for Evaluation of Text-Guided Image Edits](https://arxiv.org/abs/2506.09988) 研究人员针对性地提出了两个改进方法。一个是 “差异描述生成流水线”，通过放大编辑区域、匹配文字指令里的关键词等方式，让 AI 生成更准确的改动描述，准确率能到 75%，远超之前最好的 39%；另一个是 “瑕疵检测方法”，通过分析图片里物体的清晰度变化等数据，能准确找出 64% 的明显瑕疵，和最好的 GPT-4o 差不多。
- [2024.11] [ViBe: A Text-to-Video Benchmark for Evaluating Hallucination in Large Multimodal Models](https://arxiv.org/abs/2411.10867) 837个文本生成3782个幻觉视频。五种幻觉，包括 “物体消失”（比如人突然不见了）、“遗漏内容”（该有的东西没画）、“数量错误”（数量变多或变少）、“物体变形”（形状、大小变得奇怪）、“视觉矛盾”（出现物理上不可能的情况，比如火车突然垂直于轨道）
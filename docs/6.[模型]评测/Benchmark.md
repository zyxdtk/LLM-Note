# 大模型评测

## 一键评测

- [huggingface/lighteval](https://github.com/huggingface/lighteval)
- [open-compass/opencompass](https://github.com/open-compass/opencompass)
  - [https://rank.opencompass.org.cn/home](https://rank.opencompass.org.cn/home)
- [open-compass/VLMEvalKit](https://github.com/open-compass/VLMEvalKit)
- [modelscope/evalscope](https://github.com/modelscope/evalscope)
- [agi-eval](https://agi-eval.cn/topRanking)

## 综合基准
- chatbot-arena 包括文本、t2i、web2dev、t2v、搜索、copilot等榜单。
    - https://lmarena.ai/
    - [chatbot-arena-leaderboard](https://huggingface.co/spaces/lmarena-ai/chatbot-arena-leaderboard) 截止2025-09-18，gemini-2.5-pro排在榜一，qwen3-max-preview是榜3
- 推理&知识 
  - 人类最后考试 https://agi.safe.ai/
  - 视觉推理  https://mmmu-benchmark.github.io/
- 科学 https://github.com/idavidrein/gpqa
- 数学 https://artofproblemsolving.com/wiki/index.php/AIME_Problems_and_Solutions
- 代码  
  - 代码生成 https://livecodebench.github.io/
  - 代码编辑 https://aider.chat/docs/leaderboards/
  - Agent编程 https://www.swebench.com/
- 事实
  - https://openai.com/index/introducing-simpleqa/   https://github.com/openai/simple-evals/
- 图像理解 https://github.com/reka-ai/reka-vibe-eval
- 长上下文   
  - 多轮一致性 https://arxiv.org/html/2409.12640v2
- 多语言
  - https://huggingface.co/datasets/CohereForAI/Global-MMLU
- 

- [Open LLM Leaderboard Archived](https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard#/) 开放llm榜单，2024-10-17更新
- [2023.07] [SuperCLUE: A Comprehensive Chinese Large Language Model Benchmark](https://arxiv.org/abs/2307.15020) 中文大模型榜单
  - [CLUEbenchmark/SuperCLUE](https://github.com/CLUEbenchmark/SuperCLUE) 不更新了
  - [https://www.superclueai.com/](https://www.superclueai.com/) 2025-08还在更新

- SuperCLUE总排行榜[link]
- Text-to-Video Generation on MSR-VTT[link]
- Video Generation on UCF-101[link]


## 幻觉&TruthFull

- [2025.09] [Why Language Models Hallucinate](https://arxiv.org/abs/2509.04664) openai产生幻觉是因为模型评估鼓励模型猜而不是保持谦逊。解决办法需要重构目前的所有评估集。
- [2025.05] [HalluMix: A Task-Agnostic, Multi-Domain Benchmark for Real-World Hallucination Detection](https://arxiv.org/abs/2505.00506) 
- [2025.04] [HalluLens: LLM Hallucination Benchmark](https://arxiv.org/abs/2504.17550) meta
- [2024.11] [Measuring short-form factuality in large language models](https://arxiv.org/abs/2411.04368) openai的simpleQA,4326个问题。人工双盲标注，问答都短。多样性：话题、答案类型。答案唯一。人工标注误差3%。答错扣分。最猜答案做惩罚。
- [2024.11] [DAHL: Domain-specific Automated Hallucination Evaluation of Long-Form Text through a Benchmark Dataset in Biomedicine](https://arxiv.org/abs/2411.09255) 生物领域的幻觉
- [2024.10] [FaithBench: A Diverse Hallucination Benchmark for Summarization by Modern LLMs](https://arxiv.org/abs/2410.13210)
- [2023.12] [FActScore: Fine-grained Atomic Evaluation of Factual Precision in Long Form Text Generation](https://aclanthology.org/2023.emnlp-main.741/) meta.把长答案拆成小事实分别核对。自动评估模型误差不到2%。
- [2023.10] [Evaluating Hallucinations in Chinese Large Language Models](https://arxiv.org/abs/2310.03368) 复旦、上海ai实验室。
- [2023.05] [Do Language Models Know When They're Hallucinating References?](https://arxiv.org/abs/2305.18248) 检测的幻觉的方法：直接提问和问多次。
- [2023.05] [HaluEval: A Large-Scale Hallucination Evaluation Benchmark for Large Language Models](https://arxiv.org/abs/2305.11747) 35k幻觉数据。用chatgpt生成幻觉数据，然后人工打标挑选最好的。
  - [RUCAIBox/HaluEval](https://github.com/RUCAIBox/HaluEval)
- [2023] [AVeriTeC: A Dataset for Real-world Claim Verification with Evidence from the Web](https://proceedings.neurips.cc/paper_files/paper/2023/hash/cd86a30526cd1aff61d6f89f107634e4-Abstract-Datasets_and_Benchmarks.html)
  - [AVeriTeC Dataset(2024)](https://fever.ai/dataset/averitec.html) 4568个样本。有3个label："Supported", "Refuted", "Not Enough Evidence", "Conflicting Evidence/Cherry-picking".
- [2022.07] [Language Models (Mostly) Know What They Know](https://arxiv.org/abs/2207.05221) Anthropic。越大的模型校准度越好。给自己打分，fewshot会更准。整体来看，大型语言模型确实具备一定的 “自我认知” 能力：既能判断自己的答案对不对（P (True)），也能预判自己会不会答（P (IK)），而且模型越大、给的提示越合适，这种能力就越强。不过模型也有局限，比如容易被自己生成的答案误导，换任务时校准度会下降，且目前主要还是模仿人类知识，没法区分 “事实真相” 和 “人类常说的话”。
- [2022.05] [Teaching Models to Express Their Uncertainty in Words](https://arxiv.org/abs/2205.14334) openai,“语言化概率”（verbalized probability），就是让模型用文字或数字直接说清对答案的把握，比如 “90% 信心” 或 “高信心”。
- [2021] [TruthfulQA: Measuring How Models Mimic Human Falsehoods](https://arxiv.org/abs/2109.07958) openai, 817个问题覆盖37个领域。 针对imitative falsehoods，模型越大反而表现越差。但是通过微调可以解决这些错误。92.85% 的答案能在维基百科里找到对应的标题；近 40% 的问题需要结合多段文档推理，还有 17% 需要一些常识才能答出来。
  - [sylinrl/TruthfulQA](https://github.com/sylinrl/TruthfulQA)
- [HHEM](https://github.com/vectara/hallucination-leaderboard)
    - [vectara/leaderboard](https://huggingface.co/spaces/vectara/leaderboard)
- [2017.05] [TriviaQA: A Large Scale Distantly Supervised Challenge Dataset for Reading Comprehension](https://arxiv.org/abs/1705.03551) 14个trivia网站找到95k的qa对，平均每个q配6个证据文档。

## 角色扮演

- [2025.02] [CoSER: Coordinating LLM-Based Persona Simulation of Established Roles](https://arxiv.org/abs/2502.09082) 1.8w角色，3w对话
    - [Neph0s/COSER](https://github.com/Neph0s/COSER)
- [2024.08] [MMRole: A Comprehensive Framework for Developing and Evaluating Multimodal Role-Playing Agents](https://arxiv.org/abs/2408.04203) 多模态角色扮演评估，85 characters, 11K images, and 14K single or multi-turn dialogues
    - [YanqiDai/MMRole](https://github.com/YanqiDai/MMRole)
- [2024.01] [Large Language Models are Superpositions of All Characters: Attaining Arbitrary Role-play via Self-Alignment](https://arxiv.org/abs/2401.12474) 4k个角色，3.6w的对话
    - [OFA-Sys/Ditto](https://github.com/OFA-Sys/Ditto)
- [2024.01] [CharacterEval: A Chinese Benchmark for Role-Playing Conversational Agent Evaluation](https://arxiv.org/abs/2401.01275) 中文角色扮演评估，77个角色，1785个多轮对话
- [2023.12] [RoleEval: A Bilingual Role Evaluation Benchmark for Large Language Models](https://arxiv.org/abs/2312.16132) 中英双语角色评估，300个角色，6000个问题
- [2023.10] [RoleLLM: Benchmarking, Eliciting, and Enhancing Role-Playing Abilities of Large Language Models](https://arxiv.org/abs/2310.00746) 
    - [InteractiveNLP-Team/RoleLLM-public](https://github.com/InteractiveNLP-Team/RoleLLM-public)
- [SuperCLUE-Role: 重新定义中文角色大模型测评基准](https://github.com/CLUEbenchmark/SuperCLUE-Role)


## 多模态

- [2024.10] [MEGA-Bench: Scaling Multimodal Evaluation to over 500 Real-World Tasks](https://arxiv.org/abs/2410.10563) 505个任务，8186个样本。
    - [MEGA-Bench page](https://tiger-ai-lab.github.io/MEGA-Bench/)
    - [MEGA-Bench Leaderboard](https://huggingface.co/spaces/TIGER-Lab/MEGA-Bench)


## 电商

- [2025.05] [TransBench: Benchmarking Machine Translation for Industrial-Scale Applications](https://arxiv.org/abs/2505.14244) 评测体系中的数据集包含 “电商文化” 类别
    - [transbench](https://transbench.com/)
- [2025.02] [ChineseEcomQA: A Scalable E-commerce Concept Evaluation Benchmark for Large Language Models](https://arxiv.org/pdf/2502.20196)
- [2024.10] [Shopping MMLU: A Massive Multi-Task Online Shopping Benchmark for Large Language Models](https://arxiv.org/abs/2410.20745)

## 推理

- [2025.09] [MORABLES: A Benchmark for Assessing Abstract Moral Reasoning in LLMs with Fables](https://arxiv.org/abs/2509.12371) 道德推理能力。现在的大模型更多的还是记答案，并没有真的懂了。越大的模型越强，但是容易出现断章取义错误。ai容易自相矛盾。推理增强模型没有更好(应该是rl没有在这类人物上训练过)。ai不敢选都不对。
- [2025.05] [FABLE: A Novel Data-Flow Analysis Benchmark on Procedural Text for Large Language Model Evaluation](https://arxiv.org/html/2505.24258v1) 评估模型在程序化的数据流上的推理能力：烹饪食谱、旅行路线和自动化计划。


## 安全

- [2025.09] [detecting-and-reducing-scheming-in-ai-models](https://openai.com/index/detecting-and-reducing-scheming-in-ai-models/) 发现ai的阴谋
  - [Stress Testing Deliberative Alignment for Anti-Scheming Training](https://static1.squarespace.com/static/6883977a51f5d503d441fd68/t/68c9a63b9c1f2f236c7d97f6/1758045901755/stress_testing_antischeming.pdf)


## RAG

- [2024.09] [Fact, Fetch, and Reason: A Unified Evaluation of Retrieval-Augmented Generation](https://arxiv.org/abs/2409.12941) 模型在没检索辅助时，答对率只有 0.408；就算给一些检索到的文章，答对率也只到 0.474；但如果给全所有需要的文章，答对率能到 0.729，不过就算这样，模型在算数字、处理表格这类推理题上还是容易错。“多步检索” 方法，让模型一步步生成搜索词、找文章、补全信息，再结合示例引导模型 “按步骤思考”，结果答对率提升到了 0.66，比最初提升了 50% 以上，很接近给全资料的理想状态。


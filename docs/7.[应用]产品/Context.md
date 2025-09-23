# Context Engineer


# PromoptEngineer

## 开源

- [27.2k] [stanfordnlp/dspy](https://github.com/stanfordnlp/dspy) greate，优化prompt和模型权重。
- [12.8k] [linshenkx/prompt-optimizer](https://github.com/linshenkx/prompt-optimizer) 工程助手
    - [2025-04-06｜Prompt-Optimizer: AI 提示词优化神器全攻略](https://zhuanlan.zhihu.com/p/1892351710292332883)
- [2.7k] [Eladlev/AutoPrompt](https://github.com/Eladlev/AutoPrompt)
- [458] [AIDotNet/auto-prompt](https://github.com/AIDotNet/auto-prompt)
- [250] [auto-openai-prompter](https://github.com/hwchase17/auto-openai-prompter)
    - [Langchain创始人新项目Auto-Prompt Builder一键优化你的Prompt，再也不担心写不好Prompt了](https://developer.volcengine.com/articles/7382253806995636233)
- [42] [doganarif/promptpilot](https://github.com/doganarif/promptpilot) 主要是工程助手，没有prompt优化算法。支持prompt版本控制，ab测试。用各种model+prompt的组合来提供更新prompt。

## 产品
- [autoprompt](https://www.autoprompt.cc/)
- [阿里云百炼-prompt自动优化](https://help.aliyun.com/zh/model-studio/prompt-feedback-optimization)

## 方法

- [anthropic:prompt-improver](https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/prompt-improver)
- [Automatic Prompt Engineer (APE)](https://www.promptingguide.ai/techniques/ape)
- [Datawhale：Prompt工程](https://datawhaler.feishu.cn/wiki/MgWgw5Zpfie8h9kfx4OcUUiYnJc)

## 论文

- [2025.06] [Refract ICL: Rethinking Example Selection in the Era of Million-Token Models](https://arxiv.org/abs/2506.12346)
- [2024.10] [StablePrompt: Automatic Prompt Tuning using Reinforcement Learning for Large Language Models](https://arxiv.org/abs/2410.07652) 韩国科技院KAIST。achor model更新更慢，且只有在agent model提升超过阈值的时候才会更新。
- [2024.04] [Automatic Prompt Selection for Large Language Models](https://arxiv.org/abs/2404.02717) Deakin澳大利亚迪肯大学。对样本聚类每类一个prompt，新问题用prompt评估器挑选最适合的prompt。
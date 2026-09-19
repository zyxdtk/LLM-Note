# Ox Alpha：匿名模型身份调查

> Ox Alpha 是 2026 年 8 月突然出现在 OpenRouter 上的免费隐身模型，编码能力突出但提供商匿名。
> 截至 8 月 22 日，服务层取证强烈指向智谱 AI（Zhipu AI / Z.AI），可能是未发布的 GLM-5.3 变体，但官方未确认。
> 来源：explainx.ai、TNW、Wccftech、Business Insider、TechTimes 等公开报道，以及研究者 Chetaslua、@aitrackerbot 的取证分析。

---

## 一、时间线

```
2026.08 上旬   Ox Alpha 出现在 OpenRouter，列为 "Stealth" 提供商，免费
2026.08 中旬   在编码基准测试中登顶，引发社区关注
2026.08.21     @aitrackerbot 进行 tokenizer 和视频编码器匹配测试
2026.08.22     研究者 Chetaslua 通过畸形请求获取 Java 堆栈跟踪，锁定智谱
2026.08.23     多家媒体报道，TechTimes 披露模型保留所有 prompt 的隐私问题
```

---

## 二、模型是什么

| 参数 | 详情 |
|------|------|
| 模型 ID | `stealth/ox-alpha` |
| 上下文窗口 | 100 万 tokens |
| 最大输出 | 128K tokens |
| 输入模态 | 文本 + 图片 + 视频（多模态） |
| API 格式 | OpenAI 兼容 Chat Completions |
| 价格 | 目前免费 |
| 可用平台 | OpenRouter、OpenCode |
| 官网 | [oxalpha.io](https://oxalpha.io) |
| 推理与工具 | 支持 `reasoning.enabled`、tool definitions、tool choice |

---

## 三、谁做的：取证证据（从强到弱）

### 3.1. 服务层堆栈泄露（最强证据 — 8 月 22 日）

研究者 Chetaslua（[@chetaslua](https://x.com/chetaslua)）向 OpenCode 的 Ox Alpha 直连路由发送畸形请求——将 `top_p` 设为字符串 `"abc"` 而非数字——服务器返回 Java 堆栈跟踪，暴露内部类名：

```
com.wd.paas.api.domain.v4.chat.ChatCompletionRequest
```

该包路径直接映射到智谱的文档化 API 路由 `/api/paas/v4/chat/completions`，在 [open.bigmodel.cn](https://open.bigmodel.cn) 和 [api.z.ai](https://api.z.ai) 上均存在。

### 3.2. 错误码 1214 共享

Ox Alpha 与 Z.AI 托管的 GLM 系列模型共享相同的错误码方言（error code 1214）。DeepInfra 托管的 GLM-5.2 错误码方言不同，排除了第三方托管的可能性。

### 3.3. Tokenizer 完全匹配

@aitrackerbot 对 Ox Alpha 进行了 30 次分词器探测，30/30 全部匹配 GLM 分词器。

### 3.4. 视频编码器匹配

与 GLM-5V-Turbo 在 4 个视频上的编码器签名一致。

### 3.5. 研究者置信度

Chetaslua 给出运营层置信度 0.98。但 explainx.ai 等媒体仍标注为"未确认"。

---

## 四、被排除的猜测

| 猜测 | 排除原因 |
|------|----------|
| MiMo v2.5 | 签名不同 |
| Qwen 3.8 Max | 签名不同 |
| GLM-4.6V | 签名不同 |
| DeepInfra 托管的 GLM-5.2 | 错误码方言不同 |

注意：已发布的 GLM-5.2 被排除，说明 Ox Alpha 可能是更新的未发布版本 GLM-5.3。

---

## 五、为什么匿名发布

几种分析（均为推测，无官方说法）：

1. **压力测试**：以免费形式吸引大规模真实使用数据，验证模型在极端负载下的表现
2. **避免基准刷分争议**：匿名发布可以避免"专门针对 benchmark 优化"的指控
3. **试探海外市场**：通过 OpenRouter 匿名上线，测试海外开发者对模型质量的自然反应，不受品牌偏见影响
4. **合规考量**：可能涉及尚未完成备案或审批的模型版本

智谱、OpenRouter、OpenCode 三方均保持沉默，未做任何官方声明。

---

## 六、隐私风险

TechTimes 在 8 月 23 日报道：Ox Alpha 保留了所有 prompt 数据，用户无法得知持有这些数据的公司身份。对于涉及敏感代码或商业秘密的使用场景，这是一个需要认真考虑的风险。

---

## 七、相关链接

- [Ox Alpha 官网](https://oxalpha.io)
- [explainx.ai 详细调查报告](https://www.explainx.ai/blog/ox-alpha-what-we-know-mystery-ai-model-august-2026)
- [TNW 报道](https://thenextweb.com/news/ox-alpha-stealth-model-openrouter-anonymous-provider)
- [Wccftech 报道（指向智谱 GLM）](https://wccftech.com/a-mysterious-ai-lab-is-offering-100-trillion-free-tokens-day-for-its-ox-alpha-model-as-evidence-points-to-zhipus-unreleased-glm/)
- [Business Insider 报道](https://www.businessinsider.com/ox-alpha-ai-model-mystery-2026-8)
- [TechTimes 隐私报道](https://www.techtimes.com/articles/325244/20260823/coding-model-ox-alpha-retains-every-prompt-you-cannot-name-company-holding-them.htm)
- [officechai 各方理论汇总](https://officechai.com/ai/which-lab-is-behind-the-viral-ox-alpha-model-these-are-the-theories/)

---

## 八、与本文档的关联

- 智谱 GLM 系列参见：[CodeLLM.md](../4.text-model/CodeLLM.md)（1.2 节列有 GLM-5 论文）
- 多模态模型参见：[VLM.md](./VLM.md)
- 人物/团队全景参见：[9.people/](../9.people/) 目录

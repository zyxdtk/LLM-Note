# GPU 单卡价格与算力成本参考（2026-11）

> 整理日期：2026-11-01 | 口径：单卡，价格为市场估计值（随行情波动，±20% 正常）
> 用途：自部署成本模型的硬件参数基线（配合 `src/pricing/selfhost_h200.py` 使用）
> Azure 侧租价可用 `uv run python src/pricing/fetch_gpu_rental.py` 免登录刷新；数据源分级与踩坑见文末「数据源与采集方法」

## 主表

售价 = 单卡市场售价（千卡集采价更低）；租金 = 按需租用单卡价。`$/EFLOP` = 按租金跑满 1 EFLOP(10¹⁸) FP8 有效算力的理论成本（100% MFU，实际 ÷MFU，如 30% MFU 则 ×3.3）。

| GPU | 架构 | 显存 | 带宽 | BF16 | FP8 | FP4 | 售价$ | 租金$/h | 租金$/s | $/EFLOP(FP8) | 售价$/TFLOP(FP8) |
|:--|:--|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| B300 (Blackwell Ultra) | Blackwell | 288GB HBM3e | 8.0 TB/s | 2500 | 5000 | 10000 | 50000 [估] | 7.0 | 0.001944 | 389 | 10.0 |
| B200 | Blackwell | 192GB HBM3e | 8.0 TB/s | 2250 | 4500 | 9000 | 40000 | 5.0 | 0.001389 | 309 | 8.9 |
| H200 SXM | Hopper | 141GB HBM3e | 4.8 TB/s | 989 | 1979 | - | 30000 | 3.0 | 0.000833 | 421 | 15.2 |
| H100 SXM | Hopper | 80GB HBM3 | 3.35 TB/s | 989 | 1979 | - | 25000 | 2.5 | 0.000694 | 351 | 12.6 |
| H20 | Hopper(阉割) | 96GB HBM3 | 4.0 TB/s | 148 | 296 | - | 13000 | 1.2 | 0.000333 | 1126 | 43.9 |
| A100 80G | Ampere | 80GB HBM2e | 2.0 TB/s | 312 | - | - | 12000 | 1.2 | 0.000333 | 1068(BF16) | 38.5 |
| MI355X | CDNA4 | 288GB HBM3e | 8.0 TB/s | 2300 | 4600 | 9200 | 25000 [估] | 3.5 | 0.000972 | 211 | 5.4 |
| MI325X | CDNA3 | 256GB HBM3e | 6.0 TB/s | 1307 | 2614 | - | 20000 | 2.5 | 0.000694 | 266 | 7.6 |
| MI300X | CDNA3 | 192GB HBM3 | 5.3 TB/s | 1307 | 2614 | - | 18000 | 2.0 | 0.000556 | 213 | 6.9 |
| L40S | Ada | 48GB GDDR6 | 0.86 TB/s | 362 | 733 | - | 8000 | 1.0 | 0.000278 | 379 | 10.9 |
| RTX 5090 | Blackwell 消费级 | 32GB GDDR7 | 1.79 TB/s | 419 | 838 | 1676 | 2200 | 0.4 | 0.000111 | 133 | 2.6 |

算力单位均为 dense TFLOPS（不含 sparsity 虚标）。FP8 列空 = 不支持。`[估]` = 官方未公开单卡价，按整机/竞品推测。

## 读表要点

1. **算力代际价差 3~5 倍**：同样租 1 PFLOPS FP8 一小时，MI300X/MI355X/5090 约 $0.21~0.13，H200 要 $0.42，H20 要 $1.13。B 系/AMD 卡的纸面算力性价比碾压 Hopper。
2. **但推理不只看 FLOPS**：decode 阶段是带宽瓶颈。H20 算力被砍到 296 TFLOPS 但带宽 4.0TB/s 接近 H200——它就是为推理 decode 设计的「带宽卡」，长上下文 decode 性价比不差；prefill 重（算力瓶颈）的场景它最亏。
3. **消费级 5090 是每 FLOP 之王**（$2.6/TFLOP）但没有 NVLink、显存 32GB、数据中心合规受限——适合小模型/边缘，大模型多卡互联场景不可用。
4. **售价 vs 租金盈亏线**：H200 $30k ÷ $3/h = **1 万小时 ≈ 14 个月满负载**。利用率低于 ~50% 时租比买划算；自有电 + 二手卡 + 高利用率才考虑买。
5. **租金有量价差**：表中为按需价；包月/包年合约通常 6~8 折，竞价实例（spot）可到 5 折但会被抢占。超大规模云（AWS/GCP 官方价）通常比表中高 50%+。

## 国产卡（人民币口径）

售价均为单卡人民币参考价（万），渠道/集采价差大，±30% 正常；云租金市场不透明，仅摩尔线程 S4000 在 AutoDL 有公开时租。算力均为 dense BF16/FP16 TFLOPS（国产卡宣传常混 sparsity 口径，已尽量剥离，存疑处标注）。`¥/ZF` = 按租金跑满 10²¹ 次 BF16 浮点运算的理论成本（同主表口径，可横向对比）。

| 芯片 | 显存 | 带宽 | BF16 | FP8 | 售价¥(万) | 租金¥/h | 售价¥/T(BF16) | ¥/ZF(BF16) |
|:--|:--|--:|--:|:--|--:|--:|--:|--:|
| 昇腾 910C | 128GB HBM2e | 3.2 TB/s | 780 | 存疑 | 10~11 | - | ~135 | - |
| 昇腾 910B | 64GB HBM2e | 1.6 TB/s | 320 | - | 7~10 | 华为云实例计价 | ~265 | - |
| 昇腾 950PR (2026Q4 量产) | 144GB 自研 HiZQ | 4.0 TB/s | (FP8 1P) | 1000 | 8+ [估] | - | - | - |
| 寒武纪 MLU590 | 96GB HBM2e | 2.0 TB/s | 314~345 | 支持(口径不一) | 5.7~7 [估] | - | ~180 | - |
| 寒武纪 MLU690 (2026 量产) | 196GB HBM3 | - | >700 | 支持 | 未公开 | - | - | - |
| 昆仑芯 P800 | 96GB HBM | 2.4 TB/s | ~345 [估] | 8bit 推理 | 未公开 | 百度百舸平台 | - | - |
| 沐曦 C500 | 64GB HBM2e | 1.8 TB/s | ~250~300 [估] | - | 3.7~4.8 | - | ~150 | - |
| 摩尔线程 S5000 | 80GB | 1.6 TB/s | ~500 [估] | ~1000 [估] | 未公开 | - | - | - |
| 摩尔线程 S4000 | 48GB GDDR6 | 0.77 TB/s | 100 | - | 3~4 [估] | 1.86 (AutoDL) | ~350 | 517 |
| 海光 DCU K100-AI | 64GB HBM2e | 0.9 TB/s | 196 | - | 5.34 (零售挂牌) | - | 272 | - |
| 燧原 云燧 T21 (OAM) | 64GB HBM2e | 1.8 TB/s | ~160 [估] | - | 未公开 | - | - | - |
| 燧原 S60 (推理卡) | 48GB GDDR6 | 0.67 TB/s | ~128 [估] | - | 未公开 | - | - | - |
| 天数智芯 天垓150 | 64GB | - | ~300 [估] | - | 7.5 (零售挂牌) | - | ~250 | - |
| 天数智芯 天垓100 | 32GB HBM2 | 1.2 TB/s | 147 | - | 未公开 | - | - | - |

数据可信度：最硬的是沐曦 C500 出厂均价（IPO 问询函法定披露）与昇腾 910C 渠道报价（路透社 2026-09，受 HBM 涨价两月涨 20~50%）；规格以官方手册为准的仅 S4000/燧原/天垓100；MLU590/690 来自证券时报引券商研报；其余标 [估] 的为媒体/第三方口径。来源见文末。

### 国产卡读表要点

1. **推理性价比看 910C vs H20**（同按 BF16 dense 口径）：910C ¥10 万 / 780T ≈ ¥128/T，H20 ¥9.2 万（$13k）/ 148T ≈ ¥624/T，售价算力密度差约 5 倍；但 910C 无成熟 FP8、生态为 CANN——模型迁移与算子适配的人力成本要算进 TCO。适合政策导向/供给受限场景，纯经济账仍看具体负载。
2. **生态是隐性成本**：CUDA（英伟达/AMD 类 ROCm）之外，昇腾 CANN、摩尔线程 MUSA、寒武纪 Neuware 均需模型适配；主流开源模型官方支持度 昇腾 > 寒武纪 > 其他。自部署成本模型直接套用时，MFU/带宽利用率需按生态成熟度再打折（经验：CANN 打 8 折，其他打 6~7 折起步）。
3. **多数国产卡无 FP8**：套 `selfhost_h200.py` 时需切 BF16 口径——`weight_bytes` 1.0→2.0（权重显存翻倍、卡数增加），`fp8_flops` 换 BF16 列。
4. **租金市场不透明**：昇腾走华为云、昆仑芯走百度百舸，均为实例/集群打包价，无单卡时租；唯一公开时租的 S4000（¥1.86/h）换算 ¥517/ZFLOP，与 H200 的 $421/ZFLOP 同量级——但 100T 算力的卡租金便宜只是因为算力小，单位算力成本并不占优。
5. **供给与价格波动大**：HBM 缺货下昇腾渠道价两月涨 20~50%；集采价（运营商/银行大单）通常比表中零售/估计价低 20~30%。

## 与成本模型的对应

| 成本模型参数（selfhost_h200.py CONFIG） | 取值来源 |
|:--|:--|
| `gpu_price_per_hour` | 本表租金列（H200 当前取 3.0） |
| `hbm_gb` / `bandwidth_bps` | 本表显存/带宽列 |
| `fp8_flops` | 本表 FP8 列 |
| `bw_util` / `mfu` | 实测校准系数（0.40 / 0.30），与硬件无关 |

切换硬件（如 B200+FP4 口径）= 换一行参数。FP4 有效算力按 FP4 列代入，权重字节数 1.0→0.5。

## 局限

- 价格为 2026-11 快照，GPU 租金处于长期下行通道（B 系放量后 Hopper 租金持续走低），使用时注意时效
- 单卡售价为零售/小批量参考，千卡以上集采可低 20~30%
- 未含整机配套成本（CPU/网卡/NVLink 交换机/电力/机架），整机落地成本约为纯卡价 ×1.3~1.5

## 数据源与采集方法

回答一个问题：哪些 GPU 租赁平台的报价可以公开查、怎么查、有哪些坑。

### 平台分级

| 层级 | 平台 | 查询方式 | 认证 | 覆盖 |
|:--|:--|:--|:--|:--|
| T0 开放 API | **Azure 零售价 API** | `GET https://prices.azure.com/api/retail/prices?currencyCode=USD&$filter=...`，JSON，支持 OData filter | 免登录 | H100/H200/MI300X/GB200 等整机价，on-demand/Spot/LP 三档，分区域。超大规模云里**唯一**免登录可脚本化的 |
| T0 开放 API | OpenRouter | `/api/v1/models` + 详情页 RSC 数据 | 免登录 | LLM 牌价与端点延迟（`src/pricing/fetch_openrouter.py`） |
| T0 开放 API | HuggingFace | `/api/models/{repo}`（safetensors 总参）+ README/config（激活参数、量化格式） | 免登录 | 模型规格侧，非租价 |
| T1 页面公开·SPA | Vast.ai / RunPod / Lambda | 浏览器渲染后读表；curl 只拿到空壳 | 免登录（浏览即可） | **市场化单卡时租**（$2-3 档），本表租金列的主要对标 |
| T1 页面公开 | AutoDL | 页面直读 | 免登录 | 国内：消费级卡 + 部分国产卡时租（如 S4000 ¥1.86/h） |
| T2 需登录/合同 | AWS / GCP | AWS 免签 Pricing API 已失效（需 SigV4 签名或 GB 级 bulk JSON）；GCP Catalog API 需 API key | 需 | 超大规模云官方目录价、reserved/集采价 |
| T3 三方转述 | 国内卡渠道价 | 路透/证券时报/IPO 问询函/招采公告 | - | 下节国产卡表的来源，无实时性 |

**结论：要"可公开查 + 可脚本化"，Azure 零售价 API 是超大规模云里的独苗；市场化平台页面公开但全是 SPA，只能浏览器读表。**

### 自动化脚本（Azure 侧）

`src/pricing/fetch_gpu_rental.py`：免登录，输出 `data/pricing/raw/azure_retail_YYYY-MM-DD.json` 与 `data/pricing/processed/gpu_rental_azure_YYYY-MM-DD.{csv,md}`（sku × 计费档，整机价 + 折算单卡价）。

```bash
uv run python src/pricing/fetch_gpu_rental.py
```

### 踩坑清单（2026-11 实测）

1. Azure OData 不支持 `not(contains(...))` → BadRequest「Invalid OData parameters」，所有排除逻辑（去 Windows/Spot/噪声）必须在客户端做
2. 噪声 meter：同一 skuName 下混有数十万刀的 meter（`unitOfMeasure` 同为 "1 Hour"，reserved/许可口径，如 1645165.0），按 `price > 10000` 剔除
3. Spot / Low Priority 是独立 meterName（不是独立 priceType），要分档；实测 eastus ND96isr_H100_v5：on-demand $98.32 vs Spot $18.17 vs LP $19.66（整机 8 卡）
4. 卡数拆算：NC\* 单卡整机；ND96\* 8 卡 ÷8；NCads A100 系 24 vCPU/卡；ND128 GB200 v6 等未知封装按整机输出，不猜
5. AWS GetProducts 免密调用已失效（无签名直接 400）；bulk Price List JSON 公开但单文件 GB 级，不实用 → AWS 价只能官方目录页人工读
6. Vast/RunPod/Lambda 定价页全是 SPA：curl 拿到的是 JS 空壳（页面不小但 grep 不到 GPU 名）；Vast `/api/v0/gpu_listing/` 无 cookie 返回 `success:false`；r.jina.ai 渲染代理对这些站超时 → 市场平台价走浏览器
7. 两个反直觉但真实的现象（别当 bug 修）：Azure 上 H200 on-demand 单卡（$10.60）**低于** H100（$12.29）——H100 是 2023 年定价锁死的，H200 是 2025-26 按竞争力定的；超大规模云 **Spot ≈ 市场化平台 on-demand**（H100 Spot $2.27/卡 vs $2-3/卡）——两档市场的价差本质是"确定性溢价"，不是平台成本差

### 更新工作流

1. `uv run python src/pricing/fetch_gpu_rental.py` → 当日 Azure 三档价（秒级，免登录）
2. 市场平台价：浏览器开 vast.ai/pricing、runpod.io/pod-pricing、lambda.ai/service/gpu-pricing，人工读表（暂无自动化；若要自动化需带 cookie 调各站内部 API，另行评估）
3. 对表更新本页主表「租金$/h」列 + 顶部日期戳
4. `src/pricing/selfhost_h200.py` 的 `gpu_price_per_hour` 同步改

### 已知盲区

- H20 / B300 / MI355X / 国产卡：Azure 无此命名，国际市场平台货少，主要靠 T3 转述
- 售价（买断价）没有公开 API：只有媒体/招采/IPO 问询函口径
- Azure 区域价差大（如 ND H100 v5 澳东 $142.6 vs 美东 $98.32 整机），脚本取的是全球最低 on-demand，与"就近部署"实际成本有差距

## 国产卡数据来源

- 昇腾 910C/950 报价：[路透社 HBM 涨价报道（观察者网转载）](http://m.toutiao.com/group/7683849329102111251)、[昇腾 950/960 时间表](http://m.toutiao.com/group/7506517789307450454/)
- 寒武纪 MLU590/690：[证券时报：寒武纪万亿市值](https://www.stcn.com/article/detail/3991331.html)
- 沐曦 C500 出厂均价：[上交所 IPO 问询函](https://static.sse.com.cn/stock/disclosure/announcement/c/202508/002078_20250827_OO1E.pdf)
- 摩尔线程：[S4000 官方规格](https://docs.mthreads.com/s4000/s4000-doc-online/introduction/)、[S5000 参数首曝（36氪）](https://36kr.com/p/3680191355597316)、[AutoDL 租金](https://www.autodl.com/home)
- 燧原：[S60 产品手册](https://support.enflame-tech.com/onlinedoc_hw/5-s6x/S60/product_manual/content/source/S60_product_manual.html)、[云燧 T21 手册](https://support.enflame-tech.com/onlinedoc_hw/1-t2x/t21/product_manual/content/source/product-manual.html)
- 海光：[K100-AI 苏宁挂牌](https://m.suning.com/bigimages/12449465163.html)、K100 淘宝零售
- 昆仑芯：[P800 中标招行（官网）](https://www.kunlunxin.com/news/4469.html)；天数智芯：[天垓100 官网](https://www.iluvatar.com/productDetails?fullCode=cpjs-yj-xlxl-tg100)、天垓150 淘宝零售
- 壁仞 BR100 未列入（原旗舰已被制裁打断，在售 BR106/166 无公开报价，512T dense 为发布口径）

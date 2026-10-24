# EduJev — 学科主观题批改模型

**EduJev 1.1** 是一个面向 K-12 学科主观题的自动批改模型。给它一段「题目 + 学生作答」，它在**一次前向传播**内返回三类结构化判定，并附带经过标定的把握度。

它不生成自然语言，因此没有解析错误，也不会编造评语。

本仓库提供**推理代码、模型说明与示例**。模型权重不在此仓库中（体积 644 MB），可从 Hugging Face 或 ModelScope（魔搭）获取：

> **权重下载（Hugging Face）**：`https://huggingface.co/fuyu2009/edujev-1.1`
> ```bash
> huggingface-cli download fuyu2009/edujev-1.1 --local-dir ./model
> ```
>
> **权重下载（ModelScope / 魔搭，国内镜像）**：`https://modelscope.cn/models/fuyu2009/edujev-1.1`
> ```bash
> modelscope download fuyu2009/edujev-1.1 --local_dir ./model
> ```

---

## 这个模型是什么

它是 [Laya multilingual](https://huggingface.co/convaiinnovations/laya) 的**二次微调派生版本**，不是从零训练的模型。编码器为 ModernBERT 架构，共 **321,908,998** 参数。

> 打个比方：Laya 像一位原本做「邮件分流、工单路由」的通用判读员——擅长在一瞬间读完材料、给出结构化的分类结论，并且会老实报告自己的把握有多大。EduJev 做的事，是给这位判读员做岗位培训，把判断内容换成批改：这段话错在哪一类、该打几档分、这一步推得对不对。**能力骨架没换，换的是专业领域。**

三种判定原语（沿用 Laya 的命名）：

| 题型 | API 名称 | 含义 | 输出 |
|---|---|---|---|
| 找错因 | `choice` | 从错误类型列表中选出最匹配的一项 | 各类别概率分布 |
| 打分 | `score` | 按评分档位（0~9 档）给出整体分数 | 10 档概率分布 |
| 判断 | `noul` | 判断某个步骤 / 说法是否正确 | P(True) |

三类题型在**同一次前向**中一并计算，一份含多个小题的作业只需一次模型调用。

---

## 快速开始

```bash
# 1. 装依赖
pip install -r requirements.txt

# 2. 拉权重（约 644 MB；国内可用魔搭：modelscope download fuyu2009/edujev-1.1 --local_dir ./model）
huggingface-cli download fuyu2009/edujev-1.1 --local-dir ./model

# 3. 跑示例
python infer.py --model-dir ./model --input example.jsonl
```

输出示例（每题给出判定、把握度、完整分布与 logit 间隔）：

```json
{"id": "Biology_gapfilling_324",
 "results": {
   "total_score": {"type": "score", "label": 0, "confidence": 0.820801, "...": "..."},
   "step_1_ok":   {"type": "noul",  "label": false, "confidence": 0.942157, "...": "..."},
   "step_1_err":  {"type": "choice", "label": "逻辑推理错误", "confidence": 0.954717, "...": "..."}
 }}
```

输入格式是一条 JSON（jsonl 即每行一条）：

```json
{
  "state": "考试题目：\n13. (5 分) ...\n\n学生作答：\n...",
  "questions": {
    "total_score": { "type": "score", "instructions": "请给出整体得分档位。" },
    "step_1_ok":   { "type": "noul",  "instructions": "第 1 步是否正确？" },
    "step_1_err":  { "type": "choice", "instructions": "第 1 步的错误类型？",
                     "criteria": { "无错误": "...", "符号错误": "...", "计算错误": "..." } }
  }
}
```

`choice` 必须提供 `criteria`；`score` 与 `noul` 不需要额外字段。

---

## 实测表现

评测卷：365 条记录 / 2,394 道题（SAS-Bench 来源的中文 K-12 主观题，含人工标注的整体分、分步分与错误类型）。推理长度 `max_len=1024`。

| 指标 | 数值 |
|---|---|
| **整体准确率** | **73.48%** |
| 找错因 `choice` | 61.12%（macro F1 0.5034） |
| 打分 `score` | 54.25%（macro F1 0.2831） |
| **打分 MAE** | **1.2110 档** |
| 打分 ±1 档内 | 72.05% |
| 判断 `noul` | 85.88%（macro F1 0.8587） |

**与未微调底座的对比（同卷同构成）**：整体准确率 38.64% → 73.48%（**+34.84pp**）；打分 MAE 3.8877 → 1.2110（**−68.8%**）。

**把握度标定**（温度缩放，logit 空间）：ECE 在 choice / score / noul 上从 0.1494 / 0.1821 / 0.1400 降到 **0.1176 / 0.0608 / 0.0737**。标定温度 `[3.68488, 5.15, 6.383389]` 已写入 `rl_agent_config.json`。

**推理速度**（RTX 4050 Laptop 6 GB，batch=1，bf16）：单卷 2,364 题 **68.3 秒**（34.63 题/秒，单题 28.4 ms，p90 33.1 ms），显存峰值 **1.50 GiB**。批处理不带来吞吐收益（实测严格线性），建议保持 `batch=1`。

> 发布包的自洽性已验证：用本仓库的 `infer.py` 独立跑一遍评测卷，得到 0.7348 / 0.6112 / 0.5425 / 0.8588、MAE 1.2110，与上表逐位吻合。

---

## 文档与图表

完整的方案设计与实测过程见 [`docs/`](docs/)：

| 文件 | 内容 |
|---|---|
| [`docs/technical-report.md`](docs/technical-report.md) | 完整技术报告（五章）：问题陈述 → 方案设计 → 实测验证 → 局限说明。每个数字都标注了来源 |
| [`docs/figures.md`](docs/figures.md) | 12 张图的图题、图注、解说与注意事项 |
| [`docs/figures/`](docs/figures/) | 12 张 SVG 图源，全部由实测数据绘制，可无损缩放 |

图表不靠颜色区分——用灰度 + 填充图案 + 线型 + 标记形状四重编码，黑白打印也能读。图内只放数据与图例，解说全部在文档里。

> 说明：图中涉及「本地模型 + 云端选择性复核」的部分（图 1、6、7、9）描述的是方法与实测结果；云端调用代码未随仓库发布（涉及第三方商业 API）。本地模型可完全离线运行。

---

## 局限（请务必读完再用）

1. **长上下文会被截断。** `max_len=1024`，评测卷中 **43.65%** 的题超出该长度，平均丢 409 token。打分题受损最大：未截断 MAE 0.9514，已截断 1.4778。
2. **打分档位极不均衡。** 训练数据里 0 档占 36%、9 档占 20%，两端合计 56%。这使 macro F1（0.2831）远低于准确率（54.25%）——模型倾向押两端，中间档（尤其 7 档，仅 2.5%）识别弱。
3. **`noul` 的把握度不能用于分流。** 97.99% 的判断题把握度恰好等于 1.000，实际准确率却只有 85.88%。用置信度阈值决定「是否送人工复核」会失效；若需分流判据，请改用输出里的 `margin`（top-1 与 top-2 的 logit 间隔，与温度无关）。
4. **仅覆盖中英文 K-12 学科主观题。** 跨学段、跨学科、开放域表现未验证。
5. **不生成评语。** 只输出档位与概率，不给自然语言反馈。需要评语请另接生成模型。
6. **把握度是标定后的相对值，不等于正确率。** 换到自己的数据分布上，请重新拟合温度并复验 ECE。

---

## 许可

代码与权重均采用 **Apache License 2.0**。

- 底座 [convaiinnovations/laya](https://huggingface.co/convaiinnovations/laya)（Apache-2.0）
- 编码器 [jhu-clsp/mmBERT-base](https://huggingface.co/jhu-clsp/mmBERT-base)（Apache-2.0）
- 微调数据：SAS-Bench（Apache-2.0，附「仅限研究用途」声明）、TAL-SCQ5K（MIT）

本仓库**不包含任何训练数据**。上游归属、相对底座的修改清单与数据许可说明见 [`NOTICE`](NOTICE)。

**商用提示**：底座与代码许可清晰（允许商用），但 SAS-Bench 附有「仅限研究用途」声明。模型权重是统计产物、不含数据原文，通常不构成对数据集的再分发；但这一判断涉及具体法域，商用前请自行做合规评估。

---

## 引用

```bibtex
@misc{edujev2026,
  title  = {EduJev: A Calibrated Short-Answer Grading Model Fine-tuned from Laya},
  author = {Fu, Yu},
  year   = {2026},
  note   = {Base model: convaiinnovations/laya (Apache-2.0)},
  url    = {https://huggingface.co/fuyu2009/edujev-1.1}
}
```

底座与数据集：

```bibtex
@misc{laya2026,
  title  = {Laya: A Non-Autoregressive System 1 Decision Model},
  author = {{Convai Innovations}},
  year   = {2026},
  url    = {https://huggingface.co/convaiinnovations/laya}
}

@article{lai2025sasbench,
  title   = {SAS-Bench: A Fine-Grained Benchmark for Evaluating Short Answer
             Scoring with Large Language Models},
  author  = {Lai, Peichao and Zhang, Kexuan and Lin, Yi and Zhang, Linyihan and
             Ye, Feiyang and Yan, Jinhao and Xu, Yanwei and He, Conghui and
             Wang, Yilei and Zhang, Wentao and Cui, Bin},
  journal = {arXiv preprint arXiv:2505.07247},
  year    = {2025}
}
```

---

## 致谢

本工作建立在两项开源成果之上：Convai Innovations 的 **Laya**（提供了「一次前向、结构化输出、概率标定」的决策模型范式）与 JHU CLSP 的 **mmBERT**（提供多语言编码器骨架）。数据侧感谢 PKU-DAIR 的 **SAS-Bench** 与 **TAL-SCQ5K**。

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""EduJev 1.1 最小推理脚本。

只依赖 laya(pip) + transformers + torch + safetensors，不依赖任何项目内部代码。

用法：
    python infer.py --model-dir . --input example.jsonl
    python infer.py --model-dir . --input example.jsonl --device cpu
    python infer.py --model-dir . --input example.jsonl --no-temp   # 不做温度标定
    python infer.py --model-dir . --input example.jsonl --out result.json

输入：jsonl，每行一个对象
    {"state": "...", "questions": {"qid": {"type": "score|choice|noul",
                                          "instructions": "...",
                                          "criteria": {...}   # 仅 choice 需要
                                          }}}
输出：每行一个对象，含每题的完整分布、档位判定与把握度
"""
import argparse
import json
import math
import os
import sys

import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer

from laya.common import QTYPES, amp_dtype, build_model, build_sequence, render_options

# noul 的类别下标约定：0 = False，1 = True
NOUL_FALSE, NOUL_TRUE = 0, 1


# ------------------------------------------------------------------ 工具
def softmax(xs):
    """数值稳定的 softmax。"""
    m = max(xs)
    ex = [math.exp(v - m) for v in xs]
    s = sum(ex)
    return [v / s for v in ex]


def apply_temperature(probs, temperature=1.0):
    """温度缩放，**在 logit 空间**进行，即 softmax(z / T)。

    入参是已归一化的概率 p。因 p = softmax(z)，有 log(p) = z - C（C 为常数），
    而 softmax 对平移不变，故 softmax(log(p) / T) == softmax(z / T)，两者严格等价。

    ⚠️ 常见错误：把概率直接除以温度（z = p / T）再 softmax。那是把概率当 logit
    用，量纲错误，会把分布压平——即使 T=1 也不再恒等。
    """
    if not temperature or abs(temperature - 1.0) < 1e-12:
        return list(probs)
    logs = [math.log(max(p, 1e-12)) for p in probs]
    return softmax([v / temperature for v in logs])


def build_item(tok, cfg, state, q):
    """构造与训练同源的输入。返回 None 表示该题需跳过。"""
    qtype = q["type"]
    crit = q.get("criteria") or {}
    seq, markers = build_sequence(
        tok, state,
        {"t": qtype, "ins": q["instructions"], "crit": crit},
        cfg["max_len"], cfg["head_max_len"],
    )
    # 选项数与 marker 数不一致时，说明该题结构不被支持，跳过
    if len(markers) != len(render_options({"t": qtype, "crit": crit})):
        return None
    return {"ids": seq, "markers": markers, "qtype": QTYPES[qtype], "crit": crit}


# ------------------------------------------------------------------ 主流程
def main():
    ap = argparse.ArgumentParser(description="EduJev 1.1 推理")
    ap.add_argument("--model-dir", default=".", help="模型目录（含 model.safetensors 等）")
    ap.add_argument("--input", required=True, help="输入 jsonl")
    ap.add_argument("--out", default=None, help="输出 jsonl（默认打印到 stdout）")
    ap.add_argument("--device", default=None, help="cuda / cpu（默认有卡用卡）")
    ap.add_argument("--max-len", type=int, default=None,
                    help="覆盖 config 中的 max_len（默认用 config 里的值）")
    ap.add_argument("--no-temp", action="store_true", help="不做温度标定")
    ap.add_argument("--limit", type=int, default=0, help=">0 只处理前 N 条记录")
    args = ap.parse_args()

    md = args.model_dir

    # ---- 配置与权重 ----
    with open(os.path.join(md, "rl_agent_config.json"), encoding="utf-8") as f:
        cfg = json.load(f)
    cfg["head_max_len"] = cfg.get("head_max_len", 256)
    if args.max_len:
        cfg["max_len"] = args.max_len

    temps = cfg.get("temperature") or [1.0, 1.0, 1.0]   # 顺序：choice / score / noul
    ad = amp_dtype(cfg.get("amp_dtype", "bf16"))

    if args.device:
        dev = torch.device(args.device)
    else:
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tok = AutoTokenizer.from_pretrained(os.path.join(md, "tokenizer"))
    model = build_model(cfg, encoder_dir=os.path.join(md, "encoder"))
    model.load_state_dict(load_file(os.path.join(md, "model.safetensors")), strict=True)
    model.to(dev).eval()

    print("[EduJev] device={} max_len={} amp={} temp={}".format(
        dev, cfg["max_len"], ad, "off" if args.no_temp else temps),
        file=sys.stderr)

    # ---- 逐条推理 ----
    out_fh = open(args.out, "w", encoding="utf-8") if args.out else sys.stdout
    n_rec = n_q = n_skip = 0

    def _emit(obj):
        out_fh.write(json.dumps(obj, ensure_ascii=False) + "\n")

    try:
        with open(args.input, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if args.limit and n_rec >= args.limit:
                    break
                rec = json.loads(line)
                n_rec += 1
                state = rec.get("state", "")
                results = {}

                for qid, q in (rec.get("questions") or {}).items():
                    item = build_item(tok, cfg, state, q)
                    if item is None:
                        n_skip += 1
                        results[qid] = {"type": q.get("type"), "skipped": True,
                                        "reason": "选项结构与输入不一致"}
                        continue

                    L = len(item["ids"])
                    k = len(item["markers"])
                    with torch.no_grad(), torch.autocast(
                            dev.type if dev.type == "cuda" else "cpu", dtype=ad,
                            enabled=(dev.type == "cuda")):
                        logits, _ = model(
                            torch.tensor([item["ids"]], dtype=torch.long).to(dev),
                            torch.ones((1, L), dtype=torch.long).to(dev),
                            torch.tensor([item["markers"]], dtype=torch.long).to(dev),
                            torch.ones((1, k), dtype=torch.bool).to(dev),
                            torch.tensor([item["qtype"]], dtype=torch.long).to(dev),
                        )

                    probs = torch.softmax(logits[0, :k].float(), -1).tolist()
                    qt = item["qtype"]
                    T = 1.0 if args.no_temp else temps[qt]
                    cal = apply_temperature(probs, T)

                    pred_raw = max(range(len(probs)), key=lambda i: probs[i])
                    pred_cal = max(range(len(cal)), key=lambda i: cal[i])

                    # top-1 与 top-2 的 logit 间隔（与温度无关，可用于判断题分流）
                    lg = logits[0, :k].float()
                    if k > 1:
                        top2 = torch.topk(lg, 2).values
                        margin = round((top2[0] - top2[1]).item(), 6)
                    else:
                        margin = None

                    # 逐题型的人类可读标签
                    qtype = q["type"]
                    if qtype == "choice":
                        names = list(item["crit"].keys())
                        label = names[pred_cal] if pred_cal < len(names) else str(pred_cal)
                    elif qtype == "score":
                        label = pred_cal            # 0~9 档
                    else:                            # noul
                        label = bool(pred_cal == NOUL_TRUE)

                    results[qid] = {
                        "type": qtype,
                        "label": label,
                        "pred_index": pred_cal,
                        # 所判类别的标定后概率，即该题结论的可信度
                        "confidence": round(cal[pred_cal], 6),
                        "probs_raw": [round(p, 6) for p in probs],
                        "probs_cal": [round(p, 6) for p in cal],
                        "temperature": T,
                        "margin": margin,
                    }
                    n_q += 1

                _emit({"id": rec.get("id"), "results": results})
    finally:
        if args.out:
            out_fh.close()

    print("[EduJev] 完成：{} 条记录 / {} 题（跳过 {}）".format(n_rec, n_q, n_skip),
          file=sys.stderr)


if __name__ == "__main__":
    main()

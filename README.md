## 项目介绍

本项目面向“语言模型高效推理”的课程大作业，目标是在不改动模型参数的前提下，使推理更快、显存更省。我们基于 NVIDIA 开源的 KVPress，在其基础上适配了 Hugging Face 的 Pythia‑70M（GPT‑NeoX 架构），并提供在 PG‑19 与 WikiText 数据集上的困惑度（PPL）与加速评测脚本。

基线与评测均使用 Pythia‑70M 的无训练优化方案。Pythia‑70M 属于英语语言模型，Apache‑2.0 许可，适用于研究场景。

## 环境准备

- Python ≥ 3.10
- PyTorch ≥ 2.3.1
- Transformers ≥ 4.56
- 其他：`datasets`、`accelerate`（可选）、`pandas`（评测结果落盘可选）
- 可选：`flash-attn`（提升注意力实现性能）、`optimum-quanto`（KV 缓存量化）

安装：

```bash
pip install -e .
pip install datasets
```

建议使用 GPU 环境以获得可观的速度/显存收益，且方便记录峰值显存（CUDA）。

## 适配说明（Pythia‑70M 支持）

- 已新增对 `GPTNeoXForCausalLM` 的支持（`kvpress/presses/base_press.py:28-36`），并在钩子注册中自动识别 GPT‑NeoX 的注意力模块（`self_attn`/`attention`）与基础语言模型路径（`model.model`/`model.gpt_neox`）。
- 预 RoPE 的 Q/K 提取支持 GPT‑NeoX 的融合 `query_key_value` 线性层（`kvpress/utils.py:39-47`、`kvpress/utils.py:88-92`）。
- 管线预填充阶段跨架构调用基础语言模型，确保按压缩方案执行（`kvpress/pipeline.py:209-232`）。

这确保在 Pythia‑70M 上可以直接使用各种免训练的 KV 缓存压缩方法（如 Knorm、SnapKV、ExpectedAttention）。

## 快速验证（烟测）

运行最小示例，验证在 Pythia‑70M 上的压缩与生成：

```bash
python evaluation/smoke_pythia70m.py
```

脚本路径：`evaluation/smoke_pythia70m.py`。它会使用 `KnormPress` 在一个短上下文上进行压缩并生成少量 Token，输出简单结果。

## PPL 测试（PG‑19 / WikiText）

使用脚本：`evaluation/perplexity.py`

安装依赖：

```bash
pip install -e .
pip install datasets
```

WikiText‑103 基线（不压缩）：

```bash
python evaluation/perplexity.py --model EleutherAI/pythia-70m --dataset wikitext --subset wikitext-103-v1 --press no_press --attn_implementation eager
```

WikiText‑103 压缩示例（SnapKV，压缩比例 0.5）：

```bash
python evaluation/perplexity.py --model EleutherAI/pythia-70m --dataset wikitext --subset wikitext-103-v1 --press snapkv --compression_ratio 0.5 --attn_implementation eager
```

- 一键批跑（默认 max_new_tokens=200 ，上下文截断为 --context_limit ，默认 4096）：
  - python evaluation/perplexity.py --dataset wikitext --press all --speed_only
- 单方法示例（SnapKV，压缩率 0.5）：
  - python evaluation/perplexity.py --dataset wikitext --press snapkv --compression_ratio 0.5 --speed_only
提示

如果你的目标是“在压缩作用下”的 PPL（即压缩会不会影响质量），我已为脚本加了开关，可以选择在计算 PPL 时应用压缩。

- 新增选项： --ppl_apply_press
- 用法示例（在 PPL 阶段应用 SnapKV，压缩率 0.5）：
  - python evaluation/perplexity.py --dataset wikitext --press snapkv --compression_ratio 0.5 --ppl_apply_press
- 说明：
  - 当你指定了 --ppl_apply_press 且 --press 不是 all 时，PPL计算会在该压缩方法的上下文中执行（ compute_ppl 内部对模型前向加了压缩的钩子）。
  - 这仍然是“教师强制”的PPL，而不是先生成再评估，因为生成不会用于PPL计算。

  批量在 PPL 阶段应用压缩并评测速度：
- python evaluation/perplexity.py --dataset wikitext --press all --ppl_apply_press

- 仅统计解码阶段吞吐（避免预填充影响）：
- python evaluation/perplexity.py --dataset wikitext --press snapkv --compression_ratio 0.5 --speed_only --speed_decode_only
- 防止早停稳定吞吐：
- python evaluation/perplexity.py --dataset wikitext --press snapkv --compression_ratio 0.5 --speed_only --min_new_tokens 32
- 两者同时启用：
- python evaluation/perplexity.py --dataset wikitext --press snapkv --compression_ratio 0.5 --speed_only --speed_decode_only --min_new_tokens 32

PG‑19（超长文本，取单一样本）基线与压缩：

```bash
# 基线
python evaluation/perplexity.py --model EleutherAI/pythia-70m --dataset pg19 --sample_idx 0 --press no_press

# 压缩（Knorm，压缩比例 0.5）
python evaluation/perplexity.py --model EleutherAI/pythia-70m --dataset pg19 --sample_idx 0 \
  --press knorm --compression_ratio 0.5
```

脚本会计算并输出/保存：总 Token 数、损失与 PPL、生成吞吐（tok/s）、峰值显存（bytes）与上下文 Token 数。默认保存到 `results/perplexity/` 下的 JSON 文件。

## 加速测试（速度与显存）

同样使用 `evaluation/perplexity.py` 的压缩模式会记录吞吐与峰值显存，建议：

- 设置 `--attn_implementation eager` 以保证在通用环境下稳定；若安装了 `flash-attn` 可设为 `flash_attention_2`。
- 使用较长上下文文本以更明显地观察压缩带来的收益。

示例（SnapKV，WikiText‑103）：

```bash
python evaluation/perplexity.py --model EleutherAI/pythia-70m \
  --dataset wikitext --subset wikitext-103-v1 \
  --press snapkv --compression_ratio 0.5 --attn_implementation eager
```

## 简短报告（示例结构）

- 模型：Pythia‑70M
- 数据集与设置：WikiText‑103（test split），PG‑19（单样本 `sample_idx=0`）
- 方法：Knorm / SnapKV / ExpectedAttention（均免训练），压缩比例 0.5
- 指标：PPL（越低越好）、生成吞吐（tok/s，越高越好）、峰值显存（bytes，越低越好）
- 结果：请将 `results/perplexity/*.json` 汇总为一个表格，并在 README 中给出结论性摘要（例如“在 WikiText‑103 上，SnapKV 在保持 PPL 近似不变的同时，提升吞吐 1.3×，降低峰值显存 40%”）。

建议将汇总表与复现实验命令一起提交到你的公开 GitHub 仓库，以便可复现性评分。

## 进阶功能（可选）

- 多 GPU 推理：

```python
from transformers import pipeline
pipe = pipeline("kv-press-text-generation", model="EleutherAI/pythia-70m", device_map="auto")
```

- KV 缓存量化：如需配合压缩进一步降低显存，请安装 `optimum-quanto` 并在管线中传入 `QuantizedCache`。

## 许可与致谢

- 代码基于 Apache‑2.0 许可使用与发布。
- 致谢 NVIDIA 的 KVPress 开源工作与 Hugging Face 生态。


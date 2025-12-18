## 项目介绍

这是一个面向“语言模型高效推理”的评测套件，在不改动模型参数的前提下，让推理更快、显存更省。核心能力是基于 NVIDIA 的 KVPress，在 Hugging Face 的 Pythia 系列（GPT‑NeoX 架构）上适配并评估各种 KV 缓存压缩方法的速度、显存与困惑度（PPL）。

当前默认模型为 `EleutherAI/pythia-2.8b`，兼容 GPT‑NeoX 架构。你也可以切换到其他已支持的架构。

## 环境准备

- Python ≥ 3.10
- PyTorch ≥ 2.3.1
- Transformers ≥ 4.56
- 依赖：`datasets`
- 可选：`flash-attn`（Linux/WSL2 上提升注意力性能）、`optimum-quanto`（KV 缓存量化）

安装：

```bash
pip install -e .
pip install datasets
```

建议使用 GPU 环境（CUDA）以获得更稳定的速度与显存记录。

## 快速开始（perplexity.py）

使用脚本：`evaluation/perplexity.py`

- 单方法速度/显存（SnapKV，压缩率 0.5，仅统计速度）：

```bash
python evaluation/perplexity.py --dataset wikitext --press snapkv --compression_ratio 0.5 --speed_only
```

- 一键批跑全部已启用方法（上下文截断默认 `--context_limit 4096`，默认 `--max_new_tokens 800`）：

```bash
python evaluation/perplexity.py --dataset wikitext --press all --speed_only
```

- 在 PPL 阶段应用压缩（评估质量是否受影响）：

```bash
python evaluation/perplexity.py --dataset wikitext --press snapkv --compression_ratio 0.5 --ppl_apply_press
```

- 仅计算“压缩后”的 PPL（跳过未压缩基线）：

```bash
python evaluation/perplexity.py --dataset wikitext --press all --compression_ratio 0.7 --ppl_only_press
```

- 加速“压缩后 PPL”计算（批量损失，建议与上条组合）：

```bash
python evaluation/perplexity.py --dataset wikitext --press all --compression_ratio 0.7 --ppl_only_press --ppl_fast
```

- 单方法“压缩后 PPL”且加速：

```bash
python evaluation/perplexity.py --dataset wikitext --press snapkv --compression_ratio 0.7 --ppl_only_press --ppl_fast
```

## 关键选项

- `--press`：选择压缩方法或 `no_press`。脚本内已启用的方法包括：`snapkv`、`knorm`、`keydiff`、`pyramidkv`、`random`、`streaming_llm`、`no_press`。`--press all` 批量评测当前启用集合。
- `--compression_ratio`：压缩比例（移除的 KV 对占比）。不同方法会以各自策略应用该比例。
- `--attn_implementation`：注意力实现。`eager` 在通用环境下更稳定；`flash_attention_2` 需要安装 `flash-attn`（建议 Linux/WSL2）。
- `--context_limit`：上下文最大 Token 数（默认 4096）。这是输入截断，不是滑窗注意力。
- `--max_new_tokens`：生成长度（默认 800）。
- `--ppl_apply_press`：在 PPL 计算期间应用压缩（教师强制，不做生成）。
- `--ppl_only_press`：只计算“压缩后”的 PPL，跳过未压缩基线。适用于 `--press all` 和单方法。
- `--ppl_fast`：启用压缩后 PPL 的快速路径，按窗口批量计算损失，显著减少前向次数；数值与逐 token 路径保持一致。
- `--speed_only`：仅跑速度与显存（不计算 PPL）。
- `--speed_decode_only`：只统计“解码阶段”的耗时（不包含预填充与压缩），适合更稳定的吞吐度量。
- `--min_new_tokens`：解码前 `N` 个 token 忽略 EOS，避免早停导致吞吐不稳定。
- `--max_seq_len` 与 `--stride`：PPL 的滑窗设置。增大 `--stride`（接近 `--max_seq_len`）可减少重复预填充与压缩的成本，整体更快。

示例（稳定吞吐的设置）：

```bash
python evaluation/perplexity.py --dataset wikitext --press snapkv --compression_ratio 0.5 \
  --speed_only --speed_decode_only --min_new_tokens 32
```

## `no_press` 行为说明

- `no_press` 不做任何 KV 压缩，直接使用模型的稠密因果注意力（dense causal attention），保留完整 KV 缓存。
- `--context_limit` 仅对输入做截断（例如 4096），不是滑动窗口或窗口注意力。

## 输出结果

每次运行会在控制台打印并保存到 `results/perplexity/*.json`，字段包括：

- `model`、`dataset`、`press`、`compression_ratio`
- `loss`、`ppl`（仅在非 `--speed_only`）
- `speed_tokens_per_s`（可选择 `--speed_decode_only` 统计解码吞吐）
- `peak_mem_bytes`（峰值显存，CUDA）与 `residual_mem_bytes`（运行结束时显存占用）
- `context_tokens` 与 `context_tokens_truncated`（实际 vs 截断后上下文长度）

### 关于“压缩后 PPL”的实现与性能

- 压缩在“预填充阶段”通过 forward hook 生效，随后在增量解码上评估 `log p(token | 压缩上下文)`。
- 快速路径 `--ppl_fast` 使用教师强制的批量损失计算，减少逐 token 前向次数；与标准逐步路径数值一致。
- 代码参考：
  - 新增参数注册：`evaluation/perplexity.py:287-291`（`--ppl_only_press`、`--ppl_fast`）
  - 压缩后 PPL 快速/标准路径：`evaluation/perplexity.py:150-193`
  - `--press all` 下仅压缩 PPL 的执行分支：`evaluation/perplexity.py:358-389`

## 模型与架构支持

- 已在 GPT‑NeoX 架构适配（Pythia 系列），基础语言模型路径跨架构选择，见 `kvpress/pipeline.py:214-216`。
- 支持模型列表包含 `GPTNeoXForCausalLM`，见 `kvpress/presses/base_press.py:24-36`。

## 常见问题

- 启用 `flash_attention_2` 报错未安装：在 Windows 原生环境通常无法安装官方轮子，建议切到 Linux/WSL2 或使用 `--attn_implementation eager`。
- 吞吐波动大：启用 `--speed_decode_only` 与 `--min_new_tokens`，并适当增大 `--max_new_tokens`（例如 800）。
- PPL 计算太慢：启用 `--ppl_fast`；增大 `--stride`（如 `--max_seq_len 2048 --stride 1792`）；或使用更小的数据子集（如 `--subset wikitext-2-raw-v1`）。

## 许可与致谢

- 代码基于 Apache‑2.0 许可使用与发布。
- 致谢 NVIDIA KVPress 与 Hugging Face 生态。


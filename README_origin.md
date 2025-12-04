[![PyPI 版本](https://badge.fury.io/py/kvpress.svg)](https://badge.fury.io/py/kvpress)
[![许可证](https://img.shields.io/badge/License-Apache%202.0-green.svg)](https://opensource.org/licenses/Apache-2.0)
[![Colab 示例笔记本](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1JNvaTKuuAHrl49dYB9-mdEH_y52Ib-NP?usp=drive_link)
[![Hugging Face Space](https://img.shields.io/badge/🤗%20Hugging%20Face-Space-blue)](https://huggingface.co/spaces/nvidia/kvpress)
[![博客](https://img.shields.io/badge/🤗%20Hugging%20Face-Blog-blue)](https://huggingface.co/blog/nvidia/kvpress)
[![排行榜](https://img.shields.io/badge/🤗%20HuggingFace-Leaderboard-orange)](https://huggingface.co/spaces/nvidia/kvpress-leaderboard)
[![论文](https://img.shields.io/badge/📄%20arXiv-Paper-red)](https://arxiv.org/abs/2510.00636v1)

![kvpress](kvpress.jpg)

长上下文 LLM 的部署成本很高，原因是 Transformer 的键值（KV）缓存随上下文长度线性增长。例如，在 float16 下，Llama 3.1‑70B 处理 100 万 Token 需要约 330GB 显存。KVPress 基于 🤗 transformers 实现了多种 KV 缓存压缩方法及其评测，旨在为研究者与开发者提供一套简洁统一的实现与基准。

## 安装

```bash
pip install kvpress
```

本地开发安装（包含全部开发依赖，使用 uv）：

```bash
git clone https://github.com/NVIDIA/kvpress.git
cd kvpress
uv sync --all-groups
```
<details><summary>
高级安装设置
</summary>

可选依赖建议使用 [uv](https://docs.astral.sh/uv/) 安装：

开启 flash‑attention：

```bash
git clone https://github.com/NVIDIA/kvpress.git
cd kvpress
uv sync --extra flash-attn
```

安装评测相关依赖：

```bash
git clone https://github.com/NVIDIA/kvpress.git
cd kvpress
uv sync --extra eval
```
</details>

## 用法

KVPress 提供若干在预填充阶段压缩 KV 缓存的“Press”。每种 Press 都有一个 `compression_ratio`（压缩比例）。推荐通过自定义的 `KVPressTextGenerationPipeline` 使用，它在导入 kvpress 时会以 `kv-press-text-generation` 名称注册为 transformers 的管线，并自动处理聊天模板与分词：

```python
from transformers import pipeline
from kvpress import ExpectedAttentionPress

device = "cuda:0"
model = "meta-llama/Llama-3.1-8B-Instruct"
model_kwargs = {"attn_implementation": "flash_attention_2"}
pipe = pipeline("kv-press-text-generation", model=model, device=device, model_kwargs=model_kwargs)

context = "一个很长的上下文，需要先压缩后复用"
question = "\n关于该上下文的问题"  # 可选

press = ExpectedAttentionPress(compression_ratio=0.5)
answer = pipe(context, question=question, press=press)["answer"]
```

上述示例仅对上下文进行压缩，便于针对不同问题复用压缩后的缓存。更完整的示例可参考 [Wikipedia 演示](notebooks/wikipedia_demo.ipynb)（支持 Colab）。

<details><summary>
解码期压缩（实验性）
</summary>

默认情况下，KVPress 在预填充阶段压缩。我们提供 `DecodingPress` 包装器以在解码期周期性压缩 KV 缓存，并可选缓冲最近的隐藏态。主要参数：

- `base_press`：任意 ScorerPress（如 `KnormPress`、`CriticalKVPress`）
- `compression_interval`：压缩间隔步数（默认 10）
- `target_size`：每次压缩后目标缓存大小（默认 1024）
- `hidden_states_buffer_size`：压缩前缓冲的隐藏态数量（默认 128，有些 Press 可设为 0）

解码压缩使用目标大小而非压缩比例，即每 `compression_interval` 步压一次，自动计算到 `target_size` 的比例。

```python
from transformers import pipeline
from kvpress import KnormPress, DecodingPress

device = "cuda:0"
model = "meta-llama/Llama-3.1-8B-Instruct"
model_kwargs = {"attn_implementation": "flash_attention_2"}
pipe = pipeline("kv-press-text-generation", model=model, device=device, model_kwargs=model_kwargs)

decoding_press = DecodingPress(
    base_press=KnormPress(),
    compression_steps=10,
    token_buffer_size=512
)

context = "一个需要在生成过程中压缩的长上下文"
question = "请基于该上下文讲一个长故事"
response = pipe(context, question=question, press=decoding_press)["answer"]
```

并非所有 Press 都完全兼容解码压缩，目前主要支持 ScorerPress 作为基底。

</details>

## 可用的 Press

所有当前方法均为免训练，继承自 `BasePress`（见 `kvpress/presses/base_press.py`）。

基于打分的压缩（继承 `ScorerPress`，见 `kvpress/presses/scorer_press.py`）：

- `RandomPress`：随机打分
- `KnormPress`（论文：https://arxiv.org/abs/2406.11430）：Key 逆范数
- `SnapKVPress`（论文：https://arxiv.org/abs/2404.14469）：近期 Query 的平均注意力
- `ExpectedAttentionPress`（笔记本：notebooks/expected_attention.ipynb）：基于未来 Query 分布的期望注意力
- `StreamingLLMPress`（论文：https://arxiv.org/abs/2309.17453）：保留开头和近期 Token
- `TOVAPress`（论文：https://arxiv.org/abs/2401.06104）：最后一个 Query 的注意力（跨头平均）
- `ObservedAttentionPress`（论文：https://arxiv.org/abs/2306.14048）：预填充阶段的观测注意力
- `QFilterPress`（论文：https://arxiv.org/abs/2503.02812）：将 Key 投影到 Query 的主 SVD 分量以近似注意力
- `PyramidKVPress`（论文：https://arxiv.org/abs/2406.02069）：金字塔式分配缓存预算
- `LagKVPress`（论文：https://arxiv.org/abs/2504.04704）：利用 KV 滞后信息，免 Query、免注意力、兼容 flash‑attn
- `KeyDiffPress`（论文：https://arxiv.org/abs/2504.15364）：基于 Key 相似度淘汰
- `NonCausalAttnPress`（论文：https://arxiv.org/abs/2507.08143）：基于非因果分块注意力打分
- `LeverageScorePress`（论文：https://arxiv.org/abs/2507.08143）：近似统计杠杆分（保留 Key 空间的离群点）
- `CompactorPress`（论文：https://arxiv.org/abs/2507.08143）：在 `compression_ratio` 上融合非因果注意与杠杆分
- `CURPress`（论文：https://arxiv.org/abs/2509.15038）：基于 CUR 分解的近似杠杆分压缩

其他思路：
- `ThinKPress`（论文：https://arxiv.org/pdf/2407.21018）：按通道注意力压 Key 的维度
- `SimLayerKVPress`（论文：https://arxiv.org/abs/2410.13846）：识别“懒惰层”，对其应用 StreamingLLM
- `DuoAttentionPress`（论文：https://arxiv.org/abs/2410.10819）：将头划分为检索头与流式头
- `FinchPress`（论文：https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00716/125280）：动态窗口 + Key 重旋转，类似 SnapKV
- `KVzipPress`（论文：https://arxiv.org/abs/2505.23416）：通过上下文重建识别冗余 KV，近无损但需要多次前向

组合/包装类：
- `AdaKVPress`（论文：https://arxiv.org/abs/2407.11550）：跨头保留高分，按头压缩
- `PerLayerCompressionPress`：分层设置压缩比例（实验性）
- `ComposedPress`：串联多个 Press 的钩子
- `KeyRerotationPress`：对被剪的 Key 重新旋转以保持 RoPE 连续
- `ChunkKVPress`（论文：https://arxiv.org/abs/2502.00299）：按语义块选择保留片段
- `ChunkPress`（论文：https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00716/125280）：按分块分别压缩，提升长序列均匀性
- `CriticalKVPress` / `CriticalAdaKVPress`（论文：https://arxiv.org/abs/2502.03805）：结合 `Wo @ V` 的 L1 范数与两阶段选择
- `BlockPress`（论文：https://arxiv.org/abs/2504.15364）：分块迭代压缩
- `DecodingPress`：解码期压缩
- `PrefillDecodingPress`：同时支持预填充与解码期压缩

更多 KV 缓存压缩方法可参考：
https://github.com/October2001/Awesome-KV-Cache-Compression
https://github.com/HuangOwen/Awesome-LLM-Compression?tab=readme-ov-file#kv-cache-compression

## 评测

我们提供评测 CLI（`evaluation/evaluate.py`）以在多种长上下文基准上测试不同 Press 的表现。

- 准确率：直接在 RULER、LongBench、ZeroScrolls 等数据集上评测；结果保存在 `results/...`。
- 速度与显存：可参考 `notebooks/speed_and_memory.ipynb` ；或使用下述 PPL/加速脚本进行度量。

排行榜平均表现（RULER 4k 上下文）：

<p>
  <img src="leaderboard_plot_score.png" alt="Leaderboard">
</p>

### 在 PG‑19 与 WikiText 上进行 PPL 与加速评测

新增脚本：`evaluation/perplexity.py`

安装依赖：

```bash
pip install -e .
pip install datasets
```

WikiText‑103 PPL（基线）：

```bash
python evaluation/perplexity.py --model EleutherAI/pythia-70m \
  --dataset wikitext --subset wikitext-103-v1 --press no_press --attn_implementation eager
```

WikiText‑103 加速（压缩示例）：

```bash
python evaluation/perplexity.py --model EleutherAI/pythia-70m \
  --dataset wikitext --subset wikitext-103-v1 \
  --press snapkv --compression_ratio 0.5 --attn_implementation eager
```

PG‑19 超长文本（取单一样本）基线与加速：

```bash
# 基线
python evaluation/perplexity.py --model EleutherAI/pythia-70m --dataset pg19 --sample_idx 0 --press no_press

# 压缩（示例）
python evaluation/perplexity.py --model EleutherAI/pythia-70m --dataset pg19 --sample_idx 0 \
  --press knorm --compression_ratio 0.5
```

脚本会输出/保存 PPL、生成速度（tok/s）、峰值显存、上下文 Token 数等指标。

## 量化

支持通过 transformers 的 `QuantizedCache` 进行 KV 缓存量化（参考 HF 博文）。用法示例：

```python
from transformers import QuantizedCacheConfig, QuantoQuantizedCache

config = QuantizedCacheConfig(nbits=4)
cache = QuantoQuantizedCache(config)

pipe(..., cache=cache)
```

默认使用 `DynamicCache`（不量化）。如需使用 `QuantizedCache`，请先安装 `optimum-quanto` 等依赖。

## 贡献

欢迎贡献新方法。新增 Press 可参考 `notebooks/new_press.ipynb` 的分步教程后提交 PR。

## 引用

```bibtex
@article{devoto2025expectedattention,
  title={Expected Attention: KV Cache Compression by Estimating Attention from Future Queries Distribution},
  author={Devoto, Alessio and Jeblick, Maximilian and J{\'e}gou, Simon},
  journal={arXiv preprint arXiv:2510.00636},
  year={2025},
  url={https://arxiv.org/abs/2510.00636}
}
```

## 常见问题

<details><summary>
支持的模型有哪些？
</summary>

部分 Press 依赖具体架构（如 `ExpectedAttentionPress`、`SnapKVPress`），因此可能只在部分模型上工作。当前已测试支持：`LlamaForCausalLM`、`MistralForCausalLM`、`Phi3ForCausalLM`、`Qwen2ForCausalLM`、`Qwen3ForCausalLM`、`Gemma3ForConditionalGeneration`。本仓库已适配 `GPTNeoXForCausalLM`，可用于 Pythia‑70M。
</details>

<details><summary>
如何使用多 GPU 推理？
</summary>

KVPress 通过 [accelerate](https://huggingface.co/docs/accelerate/en/index) 支持多 GPU：

```python
pipe = pipeline("kv-press-text-generation", model=model, device_map="auto")
```

</details>

<details><summary>
压缩带来的内存与吞吐提升？
</summary>

显存占用约减少为 `compression_ratio * kv_cache_size`。由于 KV 缓存变小，解码速度通常提升。可使用 `notebooks/speed_and_memory.ipynb` 或 `evaluation/perplexity.py` 进行度量。
</details>


<details> <summary> 

### How does a press work ? </summary>

A press registers a forward hook (`press.forward_hook` method) to each attention layer during the pre-filling phase. Registration can be applied using the press as a context manager (`press.__call__` method):

```python
import torch
from transformers import AutoModelForCausalLM
from kvpress import KnormPress

device = "cuda:0"
ckpt = "meta-llama/Meta-Llama-3.1-8B-Instruct"
model = AutoModelForCausalLM.from_pretrained(ckpt).to(device)
press = KnormPress(compression_ratio=0.4)

inputs = model.dummy_inputs["input_ids"].to(device)

with torch.no_grad():
    print(model(inputs).past_key_values[0][0].shape)
    # torch.Size([3, 8, 5, 128])
    
with torch.no_grad(), press(model):
    print(model(inputs).past_key_values[0][0].shape)
    # torch.Size([3, 8, 3, 128])
```
</details>

<details><summary> 

### Why not using model.generate ? 
</summary>

In fact you can use `model.generate` with a press by using the press as a context manager:

```python
with press(model):
    outputs = model.generate(inputs)
```

However, the `generate` method does not allow to exclude the question from the compression, which would artificially favors methods such as SnapKV. Ideally, we want a compression method that works whatever comes after the context (_e.g._ for use cases such as chat or document question answering). Finally the `generate` method does not allow to provide generation for multiple questions at once.

</details>



<details><summary> 

### Can I combine compression during prefilling and decoding ? 
</summary>


Combines separate presses for prefilling and decoding phases.

**Parameters:**
- `prefilling_press`: Press used during prefill phase
- `decoding_press`: Press used during decoding phase

## Usage Examples

### Basic Decoding Compression

```python
from transformers import pipeline
from kvpress import KnormPress
from kvpress import DecodingPress

# Initialize the pipeline
device = "cuda:0"
model = "meta-llama/Llama-3.1-8B-Instruct"
model_kwargs = {"attn_implementation": "flash_attention_2"}
pipe = pipeline("kv-press-text-generation", model=model, device=device, model_kwargs=model_kwargs)

# Create a decoding press that compresses every 10 steps to 512 tokens
decoding_press = DecodingPress(
    base_press=KnormPress(),
    compression_steps=10,
    token_buffer_size=512
)

# Use with pipeline
context = "A very long text you want to compress during generation"
question = "Tell me a long story about this context"
response = pipe(context, question=question, press=decoding_press)["answer"]
```

### Combined Prefill + Decoding Compression

```python
from transformers import pipeline
from kvpress import CriticalKVPress, KnormPress
from kvpress import DecodingPress, PrefillDecodingPress

# Initialize the pipeline
device = "cuda:0"
model = "meta-llama/Llama-3.1-8B-Instruct"
model_kwargs = {"attn_implementation": "flash_attention_2"}
pipe = pipeline("kv-press-text-generation", model=model, device=device, model_kwargs=model_kwargs)

# Different strategies for prefill vs decoding
prefill_press = CriticalKVPress(KnormPress())
decoding_press = DecodingPress(
    base_press=KnormPress(compression_ratio=0.2),
    compression_steps=5,
    token_buffer_size=256
)

# Combine them
combined_press = PrefillDecodingPress(
    prefilling_press=prefill_press,
    decoding_press=decoding_press
)

context = "A very long context that will be compressed during prefill"
question = "Generate a detailed analysis that will be compressed during decoding"
response = pipe(context, question=question, press=combined_press)["answer"]
```

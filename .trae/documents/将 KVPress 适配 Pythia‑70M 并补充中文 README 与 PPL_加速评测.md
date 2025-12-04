## 总体目标
- 将现有 KVPress 代码适配 Hugging Face 的 Pythia‑70M（GPT‑NeoX 架构），在不改变模型参数的前提下实现更快、更省显存的推理。
- 翻译并改写仓库 README 为中文，保留原有结构与信息，并补充在 PG‑19、WikiText 上的 PPL 与加速评测使用说明。
- 在不训练的设定下，基于 KVPress 的“Press”方法复现并报告压缩带来的速度与显存收益。

## 仓库现状简述
- 主入口与管线：`kvpress/pipeline.py`（注册 `pipeline("kv-press-text-generation")`），预填充阶段压缩 + 贪心解码；注意力打补丁：`kvpress/attention_patch.py:90` 全局包裹 `ALL_ATTENTION_FUNCTIONS` 以支持按头屏蔽。
- Press 基类与支持模型：`kvpress/presses/base_press.py:27-34` 当前列出 Llama/Mistral/Phi3/Qwen2/Qwen3/Gemma3；钩子挂载在 `layer.self_attn`。
- 常用评测：`evaluation/evaluate.py` + `evaluation/evaluate_registry.py`，面向长上下文基准（RULER、LongBench、ZeroScrolls 等），不含通用 PPL 评测；速度/显存示例在 `notebooks/speed_and_memory.ipynb`。

## 架构支持差异与适配方案
- Pythia‑70M 使用 GPT‑NeoX 架构（Transformers 类：`GPTNeoXForCausalLM`），其层路径与注意力模块命名与 Llama 类不同：通常为 `model.gpt_neox.layers[i].attention`，注意力线性层为融合的 `query_key_value`。
- 现状问题：
  - `SUPPORTED_MODELS` 未包含 `GPTNeoXForCausalLM`（`kvpress/presses/base_press.py:27-34`）。
  - 钩子默认挂在 `layer.self_attn`，而 GPT‑NeoX 中为 `layer.attention`。
  - 预 RoPE 的 Query/Key 提取工具仅覆盖 Llama/Phi3/Qwen3/Gemma3（`kvpress/utils.py:12-95`），未覆盖 GPT‑NeoX 的融合 QKV。

## 代码改动计划
1. 扩展支持模型
- 在 `kvpress/presses/base_press.py` 中引入 `from transformers import GPTNeoXForCausalLM`，将其加入 `SUPPORTED_MODELS`。
- 在 `BasePress.__call__` 内适配层与注意力模块路径：
  - 若存在 `model.gpt_neox`，则 `language_model = model.gpt_neox`，层为 `language_model.layers`。
  - 遍历层时，检测并选取注意力模块名：优先 `self_attn`，否则 `attention`；在注册钩子前为注意力模块设置 `layer_idx = i` 并注入 `rotary_emb = language_model.rotary_emb`（保证 ExpectedAttention 等方法可用）。

2. 预 RoPE状态提取适配
- 在 `kvpress/utils.py` 增加对 `transformers.models.gpt_neox.modeling_gpt_neox.GPTNeoXAttention` 的支持：
  - `get_prerope_query_states`：通过 `module.query_key_value(hidden_states)[..., :num_heads*head_dim]` 提取 Q，并 reshape 为 `(bsz, num_heads, seq_len, head_dim)`；必要时处理 QK Norm（若存在）。
  - `get_prerope_key_states`：同理从融合投影中截取 K 段；对 `num_key_value_heads` 与 `num_attention_heads` 一致的情形按当前实现处理。

3. 鲁棒性增强
- 在 `BasePress.__call__` 注册钩子时显式设置 `attention.layer_idx = i`，避免不同架构未设置该属性时导致 `forward_hook` 取 `cache.layers[module.layer_idx]` 失败（`kvpress/presses/base_press.py:133-142`）。
- 保持注意力打补丁逻辑不变，确认对 GPT‑NeoX 也能生效（`kvpress/attention_patch.py:90-110`）。

## 评测与复现计划
1. 直接可用的评测
- 现有 `evaluation/evaluate.py` + Registry 支持多种长上下文任务的准确率评测与结果落盘；可用于演示压缩方法的通用有效性与结果复现，但不包含通用 PPL。

2. 新增通用 PPL 与加速评测脚本（无训练）
- 新建 `evaluation/perplexity.py`（CLI）：
  - 支持模型与分词器自动加载（`AutoModelForCausalLM`/`AutoTokenizer`），数据集使用 Hugging Face Datasets。
  - 数据集：`--dataset wikitext --subset wikitext-103-v1` 与 `--dataset pg19`；PG‑19 仅取单一 sample（`--sample_idx`）。
  - 评测指标：按 token 计算 NLL，报告 PPL（exp 平均负 log 概率）。
  - 加速度量：记录总用时与解码吞吐（tok/s）、以及 `torch.cuda.max_memory_allocated()` 峰值显存（启用 CUDA）。
  - 压缩配置：`--press {snapkv,knorm,...}`、`--compression_ratio 0.5`、`--attn_implementation {eager,flash_attention_2}`、`--device cuda:0`。
  - 输出：将 PPL/吞吐/峰值显存/压缩后上下文长度写入 JSON 与 CSV。
- 可选在 `evaluation/README.md` 增补该脚本使用说明与示例命令；也可在 `evaluate_registry.py` 新增一个 `perplexity` 项，但保持独立脚本更直观。

3. 示例命令（README 将给出中文示例）
- 安装与环境：`pip install -e .`、`pip install datasets`；可选 `pip install flash-attn` 与 `optimum-quanto`（量化）。
- WikiText‑103：
  - Baseline：`python evaluation/perplexity.py --model EleutherAI/pythia-70m --dataset wikitext --subset wikitext-103-v1 --press no_press --attn_implementation eager`
  - 压缩：`python evaluation/perplexity.py --model EleutherAI/pythia-70m --dataset wikitext --subset wikitext-103-v1 --press snapkv --compression_ratio 0.5 --attn_implementation eager`
- PG‑19 超长文本（单样本）：
  - Baseline：`python evaluation/perplexity.py --model EleutherAI/pythia-70m --dataset pg19 --sample_idx 0 --press no_press`
  - 压缩：`python evaluation/perplexity.py --model EleutherAI/pythia-70m --dataset pg19 --sample_idx 0 --press knorm --compression_ratio 0.5`

## README 中文改写计划
- 结构保持一致：简介、安装（含可选 flash‑attn / eval / 量化）、使用示例（`pipeline("kv-press-text-generation")`）、解码期压缩、Press 列表与来源、评测、量化、贡献、引用、FAQ（支持模型与多 GPU、收益测量）。
- 在“评测”章节补充：PPL 与加速评测脚本的运行方法、输出格式与复现实例命令；提示 PG‑19 用单样本即可。
- 在“支持模型”中明确新增 Pythia‑70M（`GPTNeoXForCausalLM`）支持并给出注意事项（例如不区分 KV 头、需设置 `attn_implementation`）。

## 验证与交付
- 适配完成后，在本地用 Pythia‑70M 小规模跑通：
  - 运行 `pipeline("kv-press-text-generation")` 的示例，确认预填充压缩与贪心解码可用。
  - 跑 WikiText/PG‑19 的 PPL 与加速脚本，基线 vs 压缩对比，生成结果 JSON/CSV 与 README 中的简短报告图表/表格。
- 交付内容：
  - 中文版 `README.md`（或 `README_zh.md` 并在仓库首页显著链接）。
  - 代码修改（`base_press.py`、`utils.py`）以支持 Pythia‑70M。
  - 新增 `evaluation/perplexity.py` 与使用文档。
  - 建议创建公开 GitHub 仓库并保留历史提交，README 内附复现实验步骤与结果。

## 风险与注意
- GPT‑NeoX 的注意力实现细节（如旋转位置编码/头数一致性）需在 Query/Key 提取处严格对齐；若 `num_key_value_heads != num_attention_heads`，按 Llama 类 `repeat_kv` 语义处理。
- `flash_attention_2` 在部分 Press 上需回退到 `eager`（例如 `ObservedAttentionPress`），评测脚本将提供参数控制。
- Pythia‑70M 属于研究用模型，英语语料，Apache‑2.0 许可；不适合作为面向用户的产品部署，用途以研究为主。
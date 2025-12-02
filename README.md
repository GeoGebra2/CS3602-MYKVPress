# MYKVPress

- 目标：在 Pythia-70M 上复现与对比多种 KV 压缩方法的加速与显存节省。
- 数据集：wikitext、pg-19。
- 运行示例：见 `scripts/run_experiments.ps1`。

## 安装
- `pip install torch transformers datasets accelerate kvpress`
- 可选安装 FlashAttention 与评估扩展。

## 硬件
- 最低：CUDA GPU 显存≥6GB，或 CPU（速度较慢）。
- 推荐：10GB–24GB 显存，可稳定运行长上下文与多压缩策略。

## 运行
- `python src/eval_ppl.py --model EleutherAI/pythia-70m --dataset wikitext --split test --stride 1024 --max_length 2048`
- `python src/speed_benchmark.py --model EleutherAI/pythia-70m --dataset pg19 --mode dense --max_new_tokens 256`
- `python src/speed_benchmark.py --model EleutherAI/pythia-70m --dataset pg19 --mode snapkv --compression_ratio 0.5 --max_new_tokens 256`
- `python src/speed_benchmark.py --model EleutherAI/pythia-70m --dataset pg19 --mode pyramidkv --compression_ratio 0.7 --max_new_tokens 256`
- `python src/speed_benchmark.py --model EleutherAI/pythia-70m --dataset pg19 --mode streaming --head_window 1024 --tail_window 2048 --max_new_tokens 256`

## 报告
- 输出 CSV 包含 PPL、tokens/s、显存峰值，对比 Dense 与压缩方法。
- 建议展示：
- `wikitext` PPL 随压缩率变化曲线
- `pg-19` 吞吐与显存随压缩率的变化
- 结论：最佳折中压缩率与方法，以及对精度的影响范围

## 结果与路径
- 速度与显存：`results/speed.csv`
- 配置示例：`configs/experiments.yaml`

## 复现
- 固定随机种子与依赖版本
- 统一通过 CLI 运行，避免手工操作

import time
import csv
import os
import sys
import argparse
import torch
from transformers import pipeline
from transformers import AutoConfig
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data import get_text_sample
from src.press_runner import create_press

def run_benchmark(model_name: str, dataset_name: str, split: str, mode: str, device: int, compression_ratio: float, head_window: int, tail_window: int, max_new_tokens: int, question: str, output_csv: str):
    use_cuda = torch.cuda.is_available() and isinstance(device, int) and device >= 0
    if use_cuda:
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats()
    if mode == "dense":
        pipe_device = device if use_cuda else -1
        pipe = pipeline("text-generation", model=model_name, device=pipe_device)
        context = get_text_sample(dataset_name, split, 0)
        start = time.time()
        out = pipe(context, max_new_tokens=max_new_tokens, do_sample=False)
        end = time.time()
    else:
        cfg = AutoConfig.from_pretrained(model_name)
        supported = {"LlamaForCausalLM", "MistralForCausalLM", "Phi3ForCausalLM", "Qwen2ForCausalLM", "Qwen3ForCausalLM", "Gemma3ForConditionalGeneration"}
        archs = cfg.architectures or []
        if not any(a in supported for a in archs):
            raise ValueError(f"unsupported kvpress model: {archs}")
        press = create_press(mode, compression_ratio, head_window, tail_window)
        pipe_device = device if use_cuda else -1
        pipe = pipeline("kv-press-text-generation", model=model_name, device=pipe_device)
        context = get_text_sample(dataset_name, split, 0)
        start = time.time()
        out = pipe(context, question=question, press=press, max_new_tokens=max_new_tokens)
        end = time.time()
    elapsed = max(end - start, 1e-6)
    tps = float(max_new_tokens) / elapsed
    mem = 0
    if use_cuda:
        mem = torch.cuda.max_memory_allocated() / (1024 ** 2)
    d = os.path.dirname(output_csv)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)
    with open(output_csv, "a", newline="") as f:
        w = csv.writer(f)
        w.writerow([model_name, dataset_name, mode, compression_ratio, max_new_tokens, round(tps, 3), round(mem, 1)])
    return tps, mem

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default="EleutherAI/pythia-70m")
    p.add_argument("--dataset", type=str, default="pg19")
    p.add_argument("--split", type=str, default="test")
    p.add_argument("--mode", type=str, default="dense")
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--compression_ratio", type=float, default=0.5)
    p.add_argument("--head_window", type=int, default=1024)
    p.add_argument("--tail_window", type=int, default=2048)
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--question", type=str, default="")
    p.add_argument("--output_csv", type=str, default="results/speed.csv")
    args = p.parse_args()
    tps, mem = run_benchmark(args.model, args.dataset, args.split, args.mode, args.device, args.compression_ratio, args.head_window, args.tail_window, args.max_new_tokens, args.question, args.output_csv)
    print(f"SPEED,{args.mode},{args.compression_ratio},{tps},{mem}")

if __name__ == "__main__":
    main()

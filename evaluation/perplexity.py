# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
import math
import os
import time
from dataclasses import asdict, dataclass

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from kvpress import (
    ExpectedAttentionPress,
    KnormPress,
    SnapKVPress,
    RandomPress,
)
from transformers import pipeline


PRESS_CHOICES = {
    "no_press": None,
    "expected_attention": ExpectedAttentionPress,
    "knorm": KnormPress,
    "snapkv": SnapKVPress,
    "random": RandomPress,
}


@dataclass
class EvalResult:
    model: str
    dataset: str
    subset: str | None
    sample_idx: int | None
    tokens: int
    loss: float | None
    ppl: float | None
    press: str | None
    compression_ratio: float | None
    attn_implementation: str | None
    speed_tokens_per_s: float | None
    peak_mem_bytes: int | None
    context_tokens: int | None


def load_text(dataset: str, subset: str | None, sample_idx: int | None) -> str:
    if dataset == "wikitext":
        subset = subset or "wikitext-103-v1"
        ds = load_dataset("wikitext", subset, split="test")
        texts = [x["text"] for x in ds]
        return "\n".join(texts)
    if dataset == "pg19":
        ds = load_dataset("pg19", split="test")
        idx = sample_idx or 0
        return ds[int(idx)]["book_text"]
    raise ValueError(f"Unsupported dataset: {dataset}")


def compute_ppl(model, tokenizer, text: str, device: str) -> tuple[float, float, int]:
    ids = tokenizer.encode(text, return_tensors="pt", add_special_tokens=False).to(device)
    if ids.size(1) < 2:
        return float("nan"), float("nan"), ids.size(1)
    with torch.no_grad():
        outputs = model(input_ids=ids, labels=ids)
        loss = float(outputs.loss.item())
        ppl = math.exp(loss)
    return loss, ppl, ids.size(1)


def measure_speed_memory(model_name: str, device: str, context: str, question: str, press_name: str | None,
                         compression_ratio: float | None, attn_impl: str | None, max_new_tokens: int = 50) -> tuple[float, int, int]:
    model_kwargs = {}
    if attn_impl:
        model_kwargs["attn_implementation"] = attn_impl
    pipe = pipeline("kv-press-text-generation", model=model_name, device=device, model_kwargs=model_kwargs)
    press = None
    if press_name and press_name in PRESS_CHOICES and PRESS_CHOICES[press_name] is not None:
        press = PRESS_CHOICES[press_name](compression_ratio=compression_ratio or 0.0)

    torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
    start = time.perf_counter()
    out = pipe(context, question=question, press=press, max_new_tokens=max_new_tokens)
    elapsed = time.perf_counter() - start
    generated = pipe.tokenizer.encode(out["answer"], add_special_tokens=False)
    tok_per_s = len(generated) / elapsed if elapsed > 0 else float("nan")
    peak_mem = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
    ctx_len = pipe.tokenizer.encode(context, add_special_tokens=False)
    return tok_per_s, peak_mem, len(ctx_len)


def main():
    parser = argparse.ArgumentParser(description="Perplexity and acceleration evaluation for Pythia-70M with KVPress")
    parser.add_argument("--model", type=str, default="EleutherAI/pythia-70m")
    parser.add_argument("--dataset", type=str, choices=["wikitext", "pg19"], required=True)
    parser.add_argument("--subset", type=str, default=None)
    parser.add_argument("--sample_idx", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--attn_implementation", type=str, default=None)
    parser.add_argument("--press", type=str, default=None, choices=list(PRESS_CHOICES.keys()))
    parser.add_argument("--compression_ratio", type=float, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=50)
    parser.add_argument("--output_dir", type=str, default="results/perplexity")
    args = parser.parse_args()

    device = args.device
    model = AutoModelForCausalLM.from_pretrained(args.model).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    text = load_text(args.dataset, args.subset, args.sample_idx)

    loss, ppl, ntoks = compute_ppl(model, tokenizer, text, device)

    speed, peak_mem, ctx_tokens = None, None, None
    if args.press is not None:
        # Use a simple question to trigger decoding and measure speed/memory
        speed, peak_mem, ctx_tokens = measure_speed_memory(
            model_name=args.model,
            device=device,
            context=text,
            question="",
            press_name=args.press,
            compression_ratio=args.compression_ratio,
            attn_impl=args.attn_implementation,
            max_new_tokens=args.max_new_tokens,
        )

    result = EvalResult(
        model=args.model,
        dataset=args.dataset,
        subset=args.subset,
        sample_idx=args.sample_idx,
        tokens=ntoks,
        loss=loss,
        ppl=ppl,
        press=args.press,
        compression_ratio=args.compression_ratio,
        attn_implementation=args.attn_implementation,
        speed_tokens_per_s=speed,
        peak_mem_bytes=peak_mem,
        context_tokens=ctx_tokens,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    stem = f"{args.dataset}__{args.subset or 'none'}__{args.model.split('/')[-1]}__{args.press or 'no_press'}"
    with open(os.path.join(args.output_dir, stem + ".json"), "w", encoding="utf-8") as f:
        json.dump(asdict(result), f, ensure_ascii=False, indent=2)

    print(json.dumps(asdict(result), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()


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
from transformers import pipeline
from kvpress import (
    AdaKVPress,
    BlockPress,
    ChunkKVPress,
    CompactorPress,
    ComposedPress,
    CriticalAdaKVPress,
    CriticalKVPress,
    DecodingPress,
    DuoAttentionPress,
    ExpectedAttentionPress,
    FinchPress,
    KeyDiffPress,
    KnormPress,
    KVzipPress,
    ObservedAttentionPress,
    PyramidKVPress,
    QFilterPress,
    RandomPress,
    SnapKVPress,
    StreamingLLMPress,
    ThinKPress,
    TOVAPress,
)


PRESS_CHOICES = {
    "adakv_expected_attention": AdaKVPress(ExpectedAttentionPress()),
    "adakv_expected_attention_e2": AdaKVPress(ExpectedAttentionPress(epsilon=1e-2)),
    "adakv_snapkv": AdaKVPress(SnapKVPress()),
    "block_keydiff": BlockPress(press=KeyDiffPress(), block_size=128),
    "chunkkv": ChunkKVPress(press=SnapKVPress(), chunk_length=20),
    "critical_adakv_expected_attention": CriticalAdaKVPress(ExpectedAttentionPress(use_vnorm=False)),
    "critical_adakv_snapkv": CriticalAdaKVPress(SnapKVPress()),
    "critical_expected_attention": CriticalKVPress(ExpectedAttentionPress(use_vnorm=False)),
    "critical_snapkv": CriticalKVPress(SnapKVPress()),
    "duo_attention": DuoAttentionPress(),
    "duo_attention_on_the_fly": DuoAttentionPress(on_the_fly_scoring=True),
    "expected_attention": ExpectedAttentionPress(),
    "finch": FinchPress(),
    "keydiff": KeyDiffPress(),
    "kvzip": KVzipPress(),
    "knorm": KnormPress(),
    "observed_attention": ObservedAttentionPress(),
    "pyramidkv": PyramidKVPress(),
    "qfilter": QFilterPress(),
    "random": RandomPress(),
    "snap_think": ComposedPress([SnapKVPress(), ThinKPress()]),
    "snapkv": SnapKVPress(),
    "streaming_llm": StreamingLLMPress(),
    "think": ThinKPress(),
    "tova": TOVAPress(),
    "compactor": CompactorPress(),
    "adakv_compactor": AdaKVPress(CompactorPress()),
    "no_press": None,
    "decoding_knorm": DecodingPress(base_press=KnormPress()),
    "decoding_streaming_llm": DecodingPress(base_press=StreamingLLMPress()),
    "decoding_tova": DecodingPress(base_press=TOVAPress()),
    "decoding_qfilter": DecodingPress(base_press=QFilterPress()),
    "decoding_adakv_expected_attention_e2": DecodingPress(base_press=AdaKVPress(ExpectedAttentionPress(epsilon=1e-2))),
    "decoding_adakv_snapkv": DecodingPress(base_press=AdaKVPress(SnapKVPress())),
    "decoding_keydiff": DecodingPress(base_press=KeyDiffPress()),
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
    residual_mem_bytes: int | None
    context_tokens: int | None
    context_tokens_truncated: int | None
    error: str | None


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


def compute_ppl(
    model,
    tokenizer,
    text: str,
    device: str,
    max_seq_len: int = 2048,
    stride: int = 512,
) -> tuple[float, float, int]:
    ids = tokenizer.encode(text, return_tensors="pt", add_special_tokens=False).to(device)
    n_toks = ids.size(1)
    if n_toks < 2:
        return float("nan"), float("nan"), n_toks

    model.eval()
    nlls = []
    total = 0
    with torch.no_grad():
        for i in range(0, n_toks, stride):
            begin = max(i + stride - max_seq_len, 0)
            end = min(i + stride, n_toks)
            trg_len = end - i
            input_ids = ids[:, begin:end]
            labels = input_ids.clone()
            if trg_len <= 0:
                continue
            labels[:, :-trg_len] = -100
            outputs = model(input_ids=input_ids, labels=labels)
            nll = outputs.loss * trg_len
            nlls.append(nll)
            total += trg_len
    loss = float(torch.stack(nlls).sum() / total) if total > 0 else float("nan")
    ppl = math.exp(loss) if not math.isnan(loss) else float("nan")
    return loss, ppl, n_toks


def measure_speed_memory(
    model_name: str,
    device: str,
    context: str,
    question: str,
    press_name: str | None,
    compression_ratio: float | None,
    attn_impl: str | None,
    max_new_tokens: int = 50,
    context_limit: int = 8192,
    answer_prefix: str | None = None,
) -> tuple[float, int, int]:
    model_kwargs = {}
    if attn_impl:
        model_kwargs["attn_implementation"] = attn_impl
    pipe = pipeline("kv-press-text-generation", model=model_name, device=device, model_kwargs=model_kwargs)
    press = None
    if press_name and press_name in PRESS_CHOICES:
        press = PRESS_CHOICES[press_name]
        if press is not None and compression_ratio is not None:
            if hasattr(press, "compression_ratio"):
                try:
                    press.compression_ratio = compression_ratio
                except Exception:
                    pass
            elif hasattr(press, "base_press") and hasattr(press.base_press, "compression_ratio"):
                try:
                    press.base_press.compression_ratio = compression_ratio
                except Exception:
                    pass

    torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
    start = time.perf_counter()
    out = pipe(
        context,
        question=question,
        press=press,
        max_new_tokens=max_new_tokens,
        max_context_length=context_limit,
        answer_prefix=answer_prefix or "",
    )
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
    parser.add_argument("--press", type=str, default=None, choices=list(PRESS_CHOICES.keys()) + ["all"])
    parser.add_argument("--compression_ratio", type=float, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=200)
    parser.add_argument("--max_seq_len", type=int, default=2048)
    parser.add_argument("--stride", type=int, default=512)
    parser.add_argument("--output_dir", type=str, default="results/perplexity")
    parser.add_argument("--context_limit", type=int, default=4096)
    parser.add_argument("--question", type=str, default="Write a concise summary of the context.")
    parser.add_argument("--answer_prefix", type=str, default="Answer: ")
    parser.add_argument("--speed_only", action="store_true")
    args = parser.parse_args()

    device = args.device
    model = AutoModelForCausalLM.from_pretrained(args.model).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    text = load_text(args.dataset, args.subset, args.sample_idx)

    if args.speed_only:
        ids = tokenizer.encode(text, return_tensors="pt", add_special_tokens=False).to(device)
        ntoks = ids.size(1)
        loss, ppl = None, None
    else:
        loss, ppl, ntoks = compute_ppl(
            model, tokenizer, text, device, max_seq_len=args.max_seq_len, stride=args.stride
        )

    speed, peak_mem, ctx_tokens = None, None, None
    residual_mem = None
    if args.speed_only or args.press is not None:
        # Support batch run across all presses
        if args.press == "all":
            os.makedirs(args.output_dir, exist_ok=True)
            for press_name in PRESS_CHOICES.keys():
                attn_impl_i = "eager" if press_name == "observed_attention" else args.attn_implementation
                try:
                    speed_i, peak_mem_i, ctx_tokens_i = measure_speed_memory(
                        model_name=args.model,
                        device=device,
                        context=text,
                        question=args.question,
                        press_name=press_name,
                        compression_ratio=args.compression_ratio,
                        attn_impl=attn_impl_i,
                        max_new_tokens=args.max_new_tokens,
                        context_limit=args.context_limit,
                        answer_prefix=args.answer_prefix,
                    )
                    residual_mem_i = int(torch.cuda.memory_allocated()) if torch.cuda.is_available() else 0
                    result_i = EvalResult(
                        model=args.model,
                        dataset=args.dataset,
                        subset=args.subset,
                        sample_idx=args.sample_idx,
                        tokens=ntoks,
                        loss=loss,
                        ppl=ppl,
                        press=press_name,
                        compression_ratio=args.compression_ratio,
                        attn_implementation=args.attn_implementation,
                        speed_tokens_per_s=speed_i,
                        peak_mem_bytes=peak_mem_i,
                        residual_mem_bytes=residual_mem_i,
                        context_tokens=ctx_tokens_i,
                        context_tokens_truncated=min(ctx_tokens_i or 0, args.context_limit),
                        error=None,
                    )
                except Exception as e:
                    result_i = EvalResult(
                        model=args.model,
                        dataset=args.dataset,
                        subset=args.subset,
                        sample_idx=args.sample_idx,
                        tokens=ntoks,
                        loss=loss,
                        ppl=ppl,
                        press=press_name,
                        compression_ratio=args.compression_ratio,
                        attn_implementation=args.attn_implementation,
                        speed_tokens_per_s=None,
                        peak_mem_bytes=None,
                        residual_mem_bytes=None,
                        context_tokens=None,
                        context_tokens_truncated=None,
                        error=str(e),
                    )
                stem = f"{args.dataset}__{args.subset or 'none'}__{args.model.split('/')[-1]}__{press_name}"
                with open(os.path.join(args.output_dir, stem + ".json"), "w", encoding="utf-8") as f:
                    json.dump(asdict(result_i), f, ensure_ascii=False, indent=2)
                print(json.dumps(asdict(result_i), ensure_ascii=False, indent=2))
            return
        # Single press path
        speed, peak_mem, ctx_tokens = measure_speed_memory(
            model_name=args.model,
            device=device,
            context=text,
            question=args.question,
            press_name=args.press,
            compression_ratio=args.compression_ratio,
            attn_impl=args.attn_implementation,
            max_new_tokens=args.max_new_tokens,
            context_limit=args.context_limit,
            answer_prefix=args.answer_prefix,
        )
        residual_mem = int(torch.cuda.memory_allocated()) if torch.cuda.is_available() else 0

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
        residual_mem_bytes=residual_mem,
        context_tokens=ctx_tokens,
        context_tokens_truncated=min(ctx_tokens or 0, args.context_limit),
        error=None,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    stem = f"{args.dataset}__{args.subset or 'none'}__{args.model.split('/')[-1]}__{args.press or 'no_press'}"
    with open(os.path.join(args.output_dir, stem + ".json"), "w", encoding="utf-8") as f:
        json.dump(asdict(result), f, ensure_ascii=False, indent=2)

    print(json.dumps(asdict(result), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()


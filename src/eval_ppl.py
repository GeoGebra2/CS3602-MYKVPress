import math
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.datasets import get_dataset

def compute_ppl(model_name: str, dataset_name: str, split: str, stride: int, max_length: int, device: str):
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    data = get_dataset(dataset_name, split)
    nll_sum = 0.0
    tok_count = 0
    for ex in data:
        text = ex["text"] if "text" in ex else (ex["content"] if "content" in ex else str(ex))
        ids = tok(text, return_tensors="pt").input_ids[0]
        i = 0
        while i < len(ids):
            end = min(i + max_length, len(ids))
            input_ids = ids[i:end].unsqueeze(0).to(device)
            labels = input_ids.clone()
            with torch.no_grad():
                out = model(input_ids, labels=labels)
            n_tokens = end - i
            nll_sum += out.loss.item() * max(n_tokens - 1, 1)
            tok_count += max(n_tokens - 1, 1)
            i += stride
    ppl = math.exp(nll_sum / max(tok_count, 1))
    return ppl

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default="EleutherAI/pythia-70m")
    p.add_argument("--dataset", type=str, default="wikitext")
    p.add_argument("--split", type=str, default="test")
    p.add_argument("--stride", type=int, default=1024)
    p.add_argument("--max_length", type=int, default=2048)
    p.add_argument("--device", type=str, default="cuda:0")
    args = p.parse_args()
    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        args.device = "cpu"
    ppl = compute_ppl(args.model, args.dataset, args.split, args.stride, args.max_length, args.device)
    print(f"PPL,{args.dataset},{args.split},{ppl}")

if __name__ == "__main__":
    main()

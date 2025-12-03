import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.custom_press import StreamingLLMPress, SnapKVPress, PyramidKVPress, RandomPress

def create_press(mode: str, compression_ratio: float = None, head_window: int = None, tail_window: int = None):
    if mode == "streaming":
        return StreamingLLMPress(head_window or 1024, tail_window or 2048)
    if mode == "snapkv":
        return SnapKVPress(compression_ratio or 0.5)
    if mode == "pyramidkv":
        return PyramidKVPress(compression_ratio or 0.5)
    if mode == "random":
        return RandomPress(compression_ratio or 0.5)
    return None

def generate_with_press(model_name: str, context: str, question: str, mode: str, device: str, compression_ratio: float, head_window: int, tail_window: int, max_new_tokens: int):
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    press = create_press(mode, compression_ratio, head_window, tail_window)
    c_ids = tok(context, return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        out = model(input_ids=c_ids, use_cache=True, output_attentions=True)
    past = out.past_key_values
    atts = out.attentions
    if press is not None:
        past = press.compress(past, {"attentions": atts})
    q_ids = None
    if question and len(question) > 0:
        q_ids = tok(question, return_tensors="pt").input_ids.to(device)
        with torch.no_grad():
            outq = model(input_ids=q_ids, use_cache=True, past_key_values=past)
        past = outq.past_key_values
    gen = []
    inp = q_ids[:, -1:] if q_ids is not None else c_ids[:, -1:]
    start = time.time()
    for _ in range(max_new_tokens):
        with torch.no_grad():
            o = model(input_ids=inp, use_cache=True, past_key_values=past)
        past = o.past_key_values
        nxt = torch.argmax(o.logits[:, -1, :], dim=-1).unsqueeze(0)
        inp = nxt
        gen.append(nxt.item())
    end = time.time()
    text = tok.decode(gen)
    return text, max(end - start, 1e-6)

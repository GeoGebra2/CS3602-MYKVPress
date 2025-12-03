import torch

class BasePress:
    def compress(self, past_key_values, extra):
        return past_key_values

def _slice_layer(k, v, idx):
    i = torch.as_tensor(idx, device=k.device)
    k = k.index_select(2, i)
    v = v.index_select(2, i)
    return k, v

def _apply_indices(past_key_values, indices):
    new_past = []
    for k, v in past_key_values:
        nk, nv = _slice_layer(k, v, indices)
        new_past.append((nk, nv))
    return tuple(new_past)

class StreamingLLMPress(BasePress):
    def __init__(self, head_window=1024, tail_window=2048):
        self.head_window = head_window
        self.tail_window = tail_window
    def compress(self, past_key_values, extra):
        k, v = past_key_values[0]
        s = k.shape[2]
        h = min(self.head_window, s)
        t = min(self.tail_window, max(s - h, 0))
        if h + t >= s:
            return past_key_values
        idx = list(range(h)) + list(range(s - t, s))
        return _apply_indices(past_key_values, idx)

class SnapKVPress(BasePress):
    def __init__(self, compression_ratio=0.5):
        self.compression_ratio = compression_ratio
    def compress(self, past_key_values, extra):
        atts = extra.get("attentions", None)
        k, v = past_key_values[0]
        s = k.shape[2]
        keep = max(int(s * self.compression_ratio), 1)
        if atts is None:
            scores = torch.norm(k[0], dim=-1).mean(dim=0)
            idx = torch.topk(scores, keep).indices.tolist()
            return _apply_indices(past_key_values, idx)
        layer_scores = []
        for a in atts:
            w = a[:, :, -1, :].mean(dim=(0,1))
            layer_scores.append(w)
        scores = torch.stack(layer_scores).mean(dim=0)
        idx = torch.topk(scores, keep).indices.tolist()
        return _apply_indices(past_key_values, idx)

class PyramidKVPress(BasePress):
    def __init__(self, compression_ratio=0.5):
        self.compression_ratio = compression_ratio
    def compress(self, past_key_values, extra):
        k, v = past_key_values[0]
        s = k.shape[2]
        kcnt = max(int(s * self.compression_ratio), 1)
        tail = max(int(kcnt * 0.5), 1)
        head_cnt = max(kcnt - tail, 0)
        tail_idx = list(range(max(s - tail, 0), s))
        head_space = max(s - tail, 1)
        step = max(head_space // max(head_cnt, 1), 1)
        head_idx = list(range(0, head_space, step))[:head_cnt]
        idx = sorted(set(head_idx + tail_idx))
        return _apply_indices(past_key_values, idx)

class RandomPress(BasePress):
    def __init__(self, compression_ratio=0.5):
        self.compression_ratio = compression_ratio
    def compress(self, past_key_values, extra):
        k, v = past_key_values[0]
        s = k.shape[2]
        keep = max(int(s * self.compression_ratio), 1)
        perm = torch.randperm(s)[:keep].tolist()
        return _apply_indices(past_key_values, perm)

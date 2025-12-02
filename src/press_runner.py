from kvpress import StreamingLLMPress, SnapKVPress, PyramidKVPress, ExpectedAttentionPress

def create_press(mode: str, compression_ratio: float = None, head_window: int = None, tail_window: int = None):
    if mode == "streaming":
        hw = head_window or 1024
        tw = tail_window or 2048
        return StreamingLLMPress(head_window=hw, tail_window=tw)
    if mode == "snapkv":
        return SnapKVPress(compression_ratio=compression_ratio or 0.5)
    if mode == "pyramidkv":
        return PyramidKVPress(compression_ratio=compression_ratio or 0.5)
    if mode == "expected":
        return ExpectedAttentionPress(compression_ratio=compression_ratio or 0.5)
    return None

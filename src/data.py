from datasets import load_dataset

def get_dataset(name: str, split: str):
    if name == "wikitext":
        return load_dataset("wikitext", "wikitext-103-v1", split=split)
    if name == "pg19":
        return load_dataset("pg19", split=split, trust_remote_code=True)
    raise ValueError("unsupported dataset")

def get_text_sample(name: str, split: str, index: int):
    ds = get_dataset(name, split)
    ex = ds[index]
    if "text" in ex:
        return ex["text"]
    if "content" in ex:
        return ex["content"]
    return str(ex)

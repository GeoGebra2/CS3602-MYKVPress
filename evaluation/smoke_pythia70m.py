# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from transformers import pipeline
from kvpress import KnormPress


def main():
    device = "cuda:0"
    model = "EleutherAI/pythia-70m"
    pipe = pipeline("kv-press-text-generation", model=model, device=device)
    context = "Pythia-70M smoke test context. " * 50
    press = KnormPress(compression_ratio=0.5)
    out = pipe(context, question="", press=press, max_new_tokens=8)
    print(out)


if __name__ == "__main__":
    main()


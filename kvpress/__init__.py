# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from kvpress.attention_patch import patch_attention_functions
from kvpress.pipeline import KVPressTextGenerationPipeline
from kvpress.presses.base_press import SUPPORTED_MODELS, BasePress
from kvpress.presses.keydiff_press import KeyDiffPress
from kvpress.presses.knorm_press import KnormPress
from kvpress.presses.random_press import RandomPress
from kvpress.presses.scorer_press import ScorerPress
from kvpress.presses.streaming_llm_press import StreamingLLMPress

# Patch the attention functions to support head-wise compression
patch_attention_functions()

__all__ = [
    "BasePress",
    "ScorerPress",
    "KnormPress",
    "KeyDiffPress",
    "RandomPress",
    "StreamingLLMPress",
    "KVPressTextGenerationPipeline",
    "SUPPORTED_MODELS",
]

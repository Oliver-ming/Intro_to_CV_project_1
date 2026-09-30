"""Unwrap generator weights from BasicSR and OpenMMLab checkpoints.

OpenMMLab's BasicVSR++ checkpoint names SpyNet convolutions and the two
upsampling layers differently from the pinned BasicSR modules. The tensor
shapes are the same, so those names are rewritten before loading.
"""

from __future__ import annotations

import re

import torch

_SPYNET = re.compile(r"^(spynet\.basic_module\.\d+\.basic_module\.)(\d+)(\.conv\.(?:weight|bias))$")
_UPSAMPLE = re.compile(r"^upsample(\d+)\.upsample_conv\.(weight|bias)$")


def generator_weights(raw: object) -> dict[str, torch.Tensor]:
    if not isinstance(raw, dict):
        raise TypeError("checkpoint must be a dict")
    state = raw
    for key in ("params_ema", "params", "state_dict"):
        value = raw.get(key)
        if isinstance(value, dict) and any(torch.is_tensor(item) for item in value.values()):
            state = value
            break
    if any(str(key).startswith("generator.") for key in state):
        state = {
            str(key)[len("generator.") :]: value
            for key, value in state.items()
            if str(key).startswith("generator.") and torch.is_tensor(value)
        }
    else:
        state = {str(key): value for key, value in state.items() if torch.is_tensor(value)}
    if not state:
        raise ValueError("checkpoint contains no generator tensors")
    return { _basicvsrpp_name(key): value for key, value in state.items() }


def _basicvsrpp_name(key: str) -> str:
    spynet = _SPYNET.match(key)
    if spynet is not None:
        return f"{spynet.group(1)}{int(spynet.group(2)) * 2}{spynet.group(3).replace('.conv', '', 1)}"
    upsample = _UPSAMPLE.match(key)
    if upsample is not None:
        return f"upconv{upsample.group(1)}.{upsample.group(2)}"
    return key

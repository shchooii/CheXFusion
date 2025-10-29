import math
from typing import Any, Union

import torch


def detach(x: Union[torch.Tensor, Any]) -> Union[torch.Tensor, Any]:
    """Detach a tensor from the computational graph"""
    if isinstance(x, torch.Tensor):
        return x.detach()
    return x


def detach_batch(batch: dict[str, Any]) -> dict[str, Any]:
    """Detach a batch from the computational graph"""
    return {k: detach(v) for k, v in batch.items()}


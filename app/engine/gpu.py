"""VRAM sufficiency check for the selected model / compute type."""

from __future__ import annotations

from typing import Optional, Union

from app.contracts import MODELS, ComputeType, GpuInfo


def vram_warning(gpu: GpuInfo, model: str, compute_type: Union[ComputeType, str]) -> Optional[str]:
    """Return a user-facing warning when gpu.total_vram_mb < MODELS[model].vram_mb[compute_type],
    else None. Suggests int8_float16 when it would fit; otherwise suggests a smaller model.
    Raises ValueError for an unknown model or compute type."""
    if model not in MODELS:
        raise ValueError(f"Unknown model: {model!r}")
    ct = ComputeType(compute_type)
    spec = MODELS[model]
    need = spec.vram_mb[ct]
    have = gpu.total_vram_mb
    if have >= need:
        return None

    msg = (
        f"Your GPU ({gpu.name}) has {have} MB of video memory, but model {model} with "
        f"{ct.value} needs about {need} MB."
    )
    int8_need = spec.vram_mb.get(ComputeType.INT8_FLOAT16)
    if ct != ComputeType.INT8_FLOAT16 and int8_need is not None and have >= int8_need:
        return msg + f" Switch the compute type to int8_float16 in Settings (needs about {int8_need} MB)."

    smaller = sorted(
        (
            (s.vram_mb[ComputeType.INT8_FLOAT16], name)
            for name, s in MODELS.items()
            if s.translate_capable and s.vram_mb[ComputeType.INT8_FLOAT16] <= have
        ),
        reverse=True,
    )
    if smaller:
        need_small, name_small = smaller[0]
        return msg + (
            f" Use the int8_float16 compute type and choose a smaller model such as {name_small} "
            f"(needs about {need_small} MB)."
        )
    if ct != ComputeType.INT8_FLOAT16 and int8_need is not None:
        return msg + (
            f" Switch the compute type to int8_float16 (needs about {int8_need} MB); "
            "loading may still fail on this GPU."
        )
    return msg + " Loading may fail on this GPU."

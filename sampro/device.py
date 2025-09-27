"""设备检测与回退工具。"""
from __future__ import annotations

import os
import warnings
from functools import lru_cache
from typing import Optional, Tuple

import torch

_ARCH_PREFIX = "sm_"


def _format_arch(major: int, minor: int) -> str:
    return f"{_ARCH_PREFIX}{major}{minor}"


@lru_cache(maxsize=1)
def _probe_cuda_support() -> Tuple[bool, Optional[str]]:
    """检查当前 CUDA GPU 是否被正在使用的 PyTorch 编译支持。"""
    if not torch.cuda.is_available():
        return False, "未检测到可用的 CUDA 设备"

    try:
        device_index = torch.cuda.current_device()
    except Exception as exc:  # pragma: no cover - 仅用于防御性兜底
        return False, f"无法获取当前 CUDA 设备: {exc}"

    try:
        major, minor = torch.cuda.get_device_capability(device_index)
    except Exception as exc:  # pragma: no cover - 仅用于防御性兜底
        return False, f"无法读取 GPU 的计算能力: {exc}"

    arch_tag = _format_arch(major, minor)
    try:
        compiled_arches = torch.cuda.get_arch_list()
    except Exception:  # pragma: no cover - 出现异常时视为未知
        compiled_arches = []

    if compiled_arches and not any(arch.startswith(arch_tag) for arch in compiled_arches):
        compiled_arches_str = ", ".join(sorted(compiled_arches)) or "<未知>"
        return (
            False,
            "当前 PyTorch 轮子未针对 GPU 架构 {arch} 编译，支持的架构包括: {compiled}".format(
                arch=arch_tag, compiled=compiled_arches_str
            ),
        )

    try:
        # 运行一个最小的 CUDA kernel，确认运行时能够在当前 GPU 上执行
        torch.empty(1, device=f"cuda:{device_index}")
    except RuntimeError as err:
        return False, f"CUDA 初始化失败: {err}"
    except Exception as exc:  # pragma: no cover - 仅用于防御性兜底
        return False, f"CUDA 初始化失败: {exc}"

    return True, arch_tag


def cuda_is_supported() -> bool:
    """返回当前 GPU 是否能被 PyTorch 使用。"""
    ok, _ = _probe_cuda_support()
    return ok


def resolve_device(preferred: str = "cuda") -> str:
    """根据运行环境选择设备，必要时自动回退到 CPU。

    Args:
        preferred: 默认首选的设备字符串，通常为 "cuda"。

    Returns:
        "cuda" 或 "cpu"。
    """
    forced = os.getenv("SAM_DEVICE")
    if forced:
        return forced

    if preferred != "cuda":
        return preferred

    ok, message = _probe_cuda_support()
    if ok:
        return "cuda"

    if message:
        warnings.warn(message + "，已自动切换到 CPU。", stacklevel=2)
    else:  # pragma: no cover - 理论上 message 不会为空
        warnings.warn("CUDA 环境不可用，已自动切换到 CPU。", stacklevel=2)
    return "cpu"

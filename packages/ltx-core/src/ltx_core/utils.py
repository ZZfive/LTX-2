from pathlib import Path
from typing import Any

import torch


def rms_norm(x: torch.Tensor, weight: torch.Tensor | None = None, eps: float = 1e-6) -> torch.Tensor:
    """Root-mean-square (RMS) normalize `x` over its last dimension.  对张量x的最后一个维度执行RMSNorm，即均方根归一化
    Thin wrapper around `torch.nn.functional.rms_norm` that infers the normalized
    shape and forwards `weight` and `eps`.
    """
    return torch.nn.functional.rms_norm(x, (x.shape[-1],), weight=weight, eps=eps)


def check_config_value(config: dict, key: str, expected: Any) -> None:  # noqa: ANN401
    actual = config.get(key)
    if actual != expected:
        raise ValueError(f"Config value {key} is {actual}, expected {expected}")


def to_velocity(
    sample: torch.Tensor,  # 当前带噪声的sample
    sigma: float | torch.Tensor,  # 当前噪声强度
    denoised_sample: torch.Tensor,  # 原始不带噪声的sample
    calc_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Convert the sample and its denoised version to velocity. 基于样本及其去噪版本计算速度。
    Returns:
        Velocity 速度
    """
    if isinstance(sigma, torch.Tensor):
        sigma = sigma.to(calc_dtype).item()
    if sigma == 0:
        raise ValueError("Sigma can't be 0.0")
    return ((sample.to(calc_dtype) - denoised_sample.to(calc_dtype)) / sigma).to(sample.dtype)  # 计算速度 $v=\frac{x_{\sigma}-x_0}{\sigma}$


def to_denoised(
    sample: torch.Tensor,  # 当前带噪声的sample
    velocity: torch.Tensor,  # 去噪速度
    sigma: float | torch.Tensor,  # 当前噪声强度
    calc_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Convert the sample and its denoising velocity to denoised sample. 基于样本及其去噪速度计算去噪样本。
    Returns:
        Denoised sample 去噪样本
    """
    if isinstance(sigma, torch.Tensor):
        sigma = sigma.to(calc_dtype)
    return (sample.to(calc_dtype) - velocity.to(calc_dtype) * sigma).to(sample.dtype)  # 计算去噪样本 $x_0=x_{\sigma}-\sigma v$


def find_matching_file(root_path: str, pattern: str) -> Path:
    """
    Recursively search for files matching a glob pattern and return the first match.  从 root_path 开始，递归搜索所有子目录，找出第一个匹配 glob 模式 pattern 的路径并返回。
    """
    matches = list(Path(root_path).rglob(pattern))
    if not matches:
        raise FileNotFoundError(f"No files matching pattern '{pattern}' found under {root_path}")
    return matches[0]

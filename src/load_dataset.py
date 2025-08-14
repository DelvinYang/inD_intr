from typing import List, Tuple
from pathlib import Path
import torch
from loguru import logger

def _parse_samples(obj) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """支持三种保存格式，返回 [(x(12,12), y(2)), ...]"""
    out = []
    # 3) {'X': Tensor, 'Y': Tensor}
    if isinstance(obj, dict) and 'X' in obj and 'Y' in obj:
        X, Y = obj['X'], obj['Y']
        if torch.is_tensor(X) and torch.is_tensor(Y) and X.ndim == 3 and X.shape[1:] == (12, 12) and Y.ndim == 2 and Y.shape[1] == 2:
            n = min(X.shape[0], Y.shape[0])
            for i in range(n):
                out.append((X[i].float(), Y[i].float()))
            return out
    # 2) {'samples': [...]}
    if isinstance(obj, dict) and 'samples' in obj:
        obj = obj['samples']
    # 1) list/tuple of (x,y)
    if isinstance(obj, (list, tuple)):
        for xy in obj:
            if isinstance(xy, (list, tuple)) and len(xy) == 2:
                x, y = xy
                x = torch.as_tensor(x).float()
                y = torch.as_tensor(y).float()
                if x.ndim == 2 and x.shape == (12, 12) and y.ndim == 1 and y.shape[0] == 2:
                    out.append((x, y))
    return out

def _collect_samples(path_like: Path, cap: int | None = None) -> Tuple[List[torch.Tensor], List[torch.Tensor], int]:
    """
    遍历 path_like（递归）收集样本：
    - cap=None 表示不截断
    - cap=int 表示最多加载 cap 条样本到内存
    返回: xs, ys, total_available
    """
    xs, ys = [], []
    total_available = 0

    if path_like.is_dir():
        all_files = [f for f in path_like.rglob("*") if f.is_file() and f.suffix.lower() == ".pt"]
        all_files.sort()
    else:
        all_files = [path_like] if path_like.suffix.lower() == ".pt" else []

    if not all_files:
        raise FileNotFoundError(f"未找到任何 .pt 文件：{path_like}")

    logger.info(f"[EvalData] Found {len(all_files)} .pt files (recursive).")

    for fp in all_files:
        if cap is not None and len(xs) >= cap:
            # 已达到 cap，不再加载后续文件
            break

        try:
            data = torch.load(fp, map_location='cpu')
        except Exception as e:
            logger.warning(f"[EvalData] 跳过 {fp.name}（无法加载）：{e}")
            continue

        samples = _parse_samples(data)
        total_available += len(samples)

        # 只有在文件实际加载了数据时才打印
        loaded_count = 0
        for x, y in samples:
            if cap is None or len(xs) < cap:
                xs.append(x.unsqueeze(0))
                ys.append(y.unsqueeze(0))
                loaded_count += 1
            else:
                break

        if loaded_count > 0:
            logger.info(f"[EvalData] Loaded {loaded_count}/{len(samples)} samples from {fp.name}")

    return xs, ys, total_available

def build_eval_dataset_small(path_like: Path, max_samples: int = 4096) -> Tuple[torch.Tensor, torch.Tensor]:
    """加载少量数据（截断到 max_samples），只打印实际加载了样本的文件"""
    xs, ys, total = _collect_samples(path_like, cap=max_samples)
    logger.info(f"[EvalData] Total available samples: {total}")
    logger.info(f"[EvalData] Loaded into memory (cap={max_samples}): {len(xs)}")
    if not xs:
        raise ValueError(f"没有从 {path_like} 解析到任何有效样本。")
    return torch.cat(xs, dim=0).contiguous(), torch.cat(ys, dim=0).contiguous()

def build_eval_dataset_full(path_like: Path) -> Tuple[torch.Tensor, torch.Tensor]:
    """加载全部数据到内存（无截断），打印所有加载了样本的文件"""
    xs, ys, total = _collect_samples(path_like, cap=None)
    logger.info(f"[EvalData] Total available samples: {total}")
    logger.info(f"[EvalData] Loaded all samples into memory: {len(xs)}")
    if not xs:
        raise ValueError(f"没有从 {path_like} 解析到任何有效样本。")
    return torch.cat(xs, dim=0).contiguous(), torch.cat(ys, dim=0).contiguous()
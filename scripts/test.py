import os
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from matplotlib import pyplot as plt
from tqdm import tqdm

from src.common import State
from src.load_dataset import build_eval_dataset_small
from src.network import DeepTemporalModel, load_model_with_pv, set_seed
from src.read_data_inD import DataReaderInD
from src.scenarioind import ScenarioInD, Vehicle


def choose_random_model(weights_root: Path,
                        percents: List[str],
                        pv_opts: List[str],
                        seed: int | None = None) -> Path:
    if seed is not None:
        random.seed(seed)

    percent = random.choice(percents)
    pv = random.choice(pv_opts)
    base_dir = weights_root / percent / pv
    if not base_dir.exists():
        raise FileNotFoundError(f"路径不存在：{base_dir}")

    subdirs = [p for p in base_dir.iterdir() if p.is_dir()]
    if not subdirs:
        raise FileNotFoundError(f"未在目录中找到任何子文件夹：{base_dir}")

    chosen_dir = random.choice(subdirs)

    pt_files = [p for p in chosen_dir.rglob("*.pt")]
    if not pt_files:
        raise FileNotFoundError(f"未在目录中找到 .pt 模型文件：{chosen_dir}")

    chosen_model = random.choice(pt_files)

    logger.info(f"[选择结果] percent={percent}, pv={pv}")
    logger.info(f"[run目录]  {chosen_dir}")
    logger.info(f"[模型文件] {chosen_model}")
    return chosen_model


def quick_evaluate(model: DeepTemporalModel,
                   dataset_path: Path,
                   device: torch.device | None = None,
                   batch_size: int = 2048) -> float:
    """
    使用数据集做一个快速 MSE 评估（只前向，不训练）。
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # TF32 / CUDA 配置（与训练脚本一致）
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    model = model.to(device)
    model.eval()

    X_cpu, Y_cpu = build_eval_dataset_small(dataset_path, max_samples=8192)
    X = X_cpu.to(device, non_blocking=True)
    Y = Y_cpu.to(device, non_blocking=True)

    criterion = nn.MSELoss(reduction='sum')
    total = 0.0
    n = X.shape[0]

    with torch.no_grad():
        # 评估阶段使用更友好的 autocast
        amp_dtype = torch.bfloat16 if device.type == 'cuda' else torch.float32
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=(device.type == 'cuda')):
            for i in range(0, n, batch_size):
                xb = X[i:i+batch_size]
                yb = Y[i:i+batch_size]
                out = model(xb)
                loss = criterion(out, yb).item()
                total += loss

    mse = total / n
    logger.info(f"[QuickEval] MSE over {n} samples: {mse:.6f}")
    return mse

# ====== VA95 评估 ======
def compute_va95_compare(model: DeepTemporalModel,
                         dataset_path: Path,
                         device: torch.device | None = None,
                         max_samples: int = 8192,
                         batch_size: int = 4096,
                         feature_index: dict | None = None,
                         use_last_step_speed: bool = True) -> tuple[float, float]:
    """
    计算并对比 VA95（||v|| * ||a|| 的 95 分位）：
      - 真值：速度来自 X，a 来自标签 Y
      - 预测：速度来自 X，a 来自 model(X)

    参数
    ----
    feature_index: 指定特征通道索引（针对 X 的每个时间步的 12 维向量）：
        {
          'vx': 0, 'vy': 1,
          'ax': 2, 'ay': 3
        }
      若与你的数据不一致，请在此处改为正确索引。
    use_last_step_speed: True 用最后一个时间步的速度；False 用 12 个时间步速度的均值
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if feature_index is None:
        feature_index = {'vx': 0, 'vy': 1, 'ax': 2, 'ay': 3}

    # 载入少量数据（可改成 full 版本）
    X_cpu, Y_cpu = build_eval_dataset_small(dataset_path, max_samples=max_samples)
    N = X_cpu.shape[0]
    logger.info(f"[VA95] Using {N} samples for VA95 computation")

    # 取速度
    # X shape: [N, 12 (T), 12 (F)]
    vx_idx, vy_idx = feature_index['vx'], feature_index['vy']
    if use_last_step_speed:
        # 取最后一步 t = -1 的速度
        vx = X_cpu[:, -1, vx_idx]
        vy = X_cpu[:, -1, vy_idx]
    else:
        # 所有 12 步速度的均值
        vx = X_cpu[:, :, vx_idx].mean(dim=1)
        vy = X_cpu[:, :, vy_idx].mean(dim=1)
    v_mag = torch.sqrt(vx**2 + vy**2)  # [N]

    # 真值加速度模长
    ay_idx = feature_index.get('ay', 1)  # 仅用于一致性显示，不直接用；Y 已是 [ax, ay]
    a_true_mag = torch.sqrt(Y_cpu[:, 0]**2 + Y_cpu[:, 1]**2)  # [N]

    # 预测加速度模长（批量前向）
    model = model.to(device).eval()
    X = X_cpu.to(device, non_blocking=True)
    pred_ax_ay = torch.empty((N, 2), dtype=torch.float32, device=device)

    with torch.no_grad():
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        amp_dtype = torch.bfloat16 if device.type == 'cuda' else torch.float32
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=(device.type == 'cuda')):
            for i in range(0, N, batch_size):
                xb = X[i:i+batch_size]
                out = model(xb)  # 期望 [B, 2] => [ax, ay]
                pred_ax_ay[i:i+batch_size] = out

    pred_ax_ay = pred_ax_ay.float().cpu()
    a_pred_mag = torch.sqrt(pred_ax_ay[:, 0]**2 + pred_ax_ay[:, 1]**2)  # [N]

    # 计算 VA = ||v|| * ||a||
    va_true = (v_mag * a_true_mag).cpu().numpy()
    va_pred = (v_mag * a_pred_mag.cpu()).numpy()

    # 95 分位
    va95_true = float(np.percentile(va_true, 95))
    va95_pred = float(np.percentile(va_pred, 95))

    logger.info(f"[VA95] True  VA95 = {va95_true:.6f}")
    logger.info(f"[VA95] Pred  VA95 = {va95_pred:.6f}")
    logger.info(f"[VA95] True  median={np.median(va_true):.6f}, mean={va_true.mean():.6f}")
    logger.info(f"[VA95] Pred  median={np.median(va_pred):.6f}, mean={va_pred.mean():.6f}")

    return va95_true, va95_pred

# =================== 主流程 ===================
if __name__ == "__main__":
    # ---------- 数据与权重路径 ----------
    prefix_number, data_path = '00', '/Users/delvin/Desktop/programs/跨文化返修/inD'
    percents = ["p100", "p90", "p80", "p70", "p60", "p50", "p40", "p30", "p20", "p10", "p05", "p0025"]
    pv_options = ["None", "sinD", "CitySim"]  # inD 的 pv 是训练时用于 inD 数据的，模型目录若包含 "inD" 会自动识别
    weights_path = Path("/Users/delvin/Desktop/programs/跨文化返修/inD_trained_weights")
    dataset_path = Path('/Users/delvin/PycharmProjects/inD_intr/inD_Dataset')  # 目录或单个 .pt 文件

    # ---------- 固定随机性并输出 seed ----------
    seed = random.randint(0, 2**32 - 1)
    set_seed(seed)
    logger.info(f"Using seed: {seed}")

    # ---------- 初始化场景（若后续需要做场景级别测试，可使用） ----------
    data_reader = DataReaderInD(prefix_number, data_path)
    scenario = ScenarioInD(data_reader)

    # ---------- 随机选择模型 ----------
    chosen_model_path = choose_random_model(weights_path, percents, pv_options, seed=seed)
    print(chosen_model_path)

    # ---------- 加载模型（自动处理 pv 与权重前缀） ----------
    model = load_model_with_pv(chosen_model_path)

    # ---------- 设备 ----------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # ---------- 快速评估 ----------
    # try:
    #     mse = quick_evaluate(model, dataset_path, device=device, batch_size=4096)
    #     logger.info(f"[Result] Quick MSE = {mse:.6f}")
    # except Exception as e:
    #     logger.error(f"Quick evaluation failed: {e}")
    #     # 至少做一个干跑（dummy forward）验证模型结构
    #     with torch.no_grad():
    #         dummy = torch.zeros(1, 12, 12, dtype=torch.float32, device=device)
    #         out = model.to(device)(dummy)
    #         logger.info(f"[DryRun] Model forward OK -> out.shape={tuple(out.shape)}")

    # ---------- 评估VA95 ----------
    try:
        va95_true, va95_pred = compute_va95_compare(
            model=model,
            dataset_path=dataset_path,
            device=device,
            max_samples=8192,
            batch_size=4096,
            feature_index={'vx': 0, 'vy': 2, 'ax': 1, 'ay': 3},  # ← 如与数据不符请改这里
            use_last_step_speed=True
        )
        logger.info(f"[Result] VA95 -> true: {va95_true:.6f} | pred: {va95_pred:.6f}")
    except Exception as e:
        logger.error(f"VA95 computation failed: {e}")
import random
from pathlib import Path
from collections import deque
from typing import Dict, Optional, List

import numpy as np
import torch
from loguru import logger
import matplotlib.pyplot as plt

from scripts.test import choose_random_model
from src.network import set_seed, load_model_with_pv
from src.read_data_inD import DataReaderInD
from src.scenarioind import ScenarioInD, Vehicle
from src.load_dataset import extract_vehicle_frame_pairs  # (scenario, ego_id, min_frames) -> List[(1x12, 1x2)] or None


@torch.no_grad()
def test_predict_single_vehicle_openloop_with_pairs(
    model: torch.nn.Module,
    scenario: ScenarioInD,
    ego_id: int | str,
    device: Optional[torch.device] = None,
    min_frames: int = 13,
) -> Optional[Dict[str, torch.Tensor]]:
    """
    单车开环（teacher forcing）评估：基于 extract_vehicle_frame_pairs 返回的逐帧特征/标签。
      - 历史与后续输入都使用“真实”的 [vx, ax, vy, ay] 与八方向距离（库函数产出）；
      - 不把模型预测的 ax, ay 回填输入；
      - 用前12帧作为历史，从第13帧开始逐帧预测该帧 [ax, ay] 并与真值对比。
    返回 None 表示帧不足 min_frames 或提取失败。
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) 取该车逐帧 (feature[1,12], label[1,2])，若中途断帧或异常，函数会返回 None
    pairs = extract_vehicle_frame_pairs(scenario, ego_id, min_frames=min_frames)
    if pairs is None or len(pairs) < min_frames:
        logger.info(f"[ego {ego_id}] insufficient frames or extraction failed; got {0 if pairs is None else len(pairs)}")
        return None

    # 2) 拼成连续张量：X_all: [N,12], Y_all: [N,2]
    X_all = torch.cat([f for (f, _) in pairs], dim=0)  # 每个 f 是 [1,12]
    Y_all = torch.cat([y for (_, y) in pairs], dim=0)  # 每个 y 是 [1, 2]
    N = X_all.shape[0]

    model = model.to(device).eval()

    # 3) 初始化历史窗口（前12帧真实特征）
    hist_seq: deque[torch.Tensor] = deque(maxlen=12)
    for i in range(12):
        xi = X_all[i]                 # [12]
        hist_seq.append(xi.detach())  # 存张量，避免后续重复构造

    # 4) 逐帧预测（从第12索引开始，对应第13帧）
    pred_list: List[List[float]] = []
    true_list: List[List[float]] = []
    frames: List[int] = []

    for t in range(12, N):  # 预测第 t 帧的 [ax, ay]，输入为 [t-12, ..., t-1] 的 12 帧特征
        # 4.1 组装输入 X: [1, 12, 12]（无不必要拷贝）
        X = torch.stack(list(hist_seq), dim=0).unsqueeze(0).to(device, dtype=torch.float32)

        # 4.2 预测该帧加速度
        a_pred = model(X).squeeze(0).float().cpu()  # [2]
        ax_pred, ay_pred = float(a_pred[0]), float(a_pred[1])

        # 4.3 真值
        ax_true, ay_true = float(Y_all[t, 0]), float(Y_all[t, 1])

        pred_list.append([ax_pred, ay_pred])
        true_list.append([ax_true, ay_true])
        frames.append(t)

        # 4.4 Teacher forcing：把“当前帧真实特征”入队，作为下一步历史的最新一帧
        hist_seq.append(X_all[t].detach())

    # 5) 指标统计
    a_pred_t = torch.tensor(pred_list, dtype=torch.float32)  # [M,2]
    a_true_t = torch.tensor(true_list, dtype=torch.float32)  # [M,2]
    mae = torch.mean(torch.abs(a_pred_t - a_true_t), dim=0)
    mse = torch.mean((a_pred_t - a_true_t) ** 2, dim=0)
    rmse = torch.sqrt(mse)

    logger.info(f"[ego {ego_id} | openloop+pairs] steps={len(frames)} "
                f"MAE(ax,ay)=({mae[0]:.6f},{mae[1]:.6f}) "
                f"RMSE(ax,ay)=({rmse[0]:.6f},{rmse[1]:.6f})")

    return {
        "frames": torch.tensor(frames, dtype=torch.long),
        "a_pred": a_pred_t,
        "a_true": a_true_t,
        "mae": mae, "rmse": rmse, "mse": mse,
    }


# ========= 入口 =========
if __name__ == "__main__":
    # ---------- 数据与权重路径 ----------
    prefix_number, data_path = '00', '/Users/delvin/Desktop/programs/跨文化返修/inD'
    percents = ["p100", "p90", "p80", "p70", "p60", "p50", "p40", "p30", "p20", "p10", "p05", "p0025"]
    pv_options = ["None", "sinD", "CitySim"]
    weights_path = Path("/Users/delvin/Desktop/programs/跨文化返修/inD_trained_weights")

    # ---------- 固定随机性并输出 seed ----------
    seed = random.randint(0, 2**32 - 1)
    set_seed(seed)
    logger.info(f"Using seed: {seed}")

    # ---------- 初始化场景 ----------
    data_reader = DataReaderInD(prefix_number, data_path)
    scenario = ScenarioInD(data_reader)

    # ---------- 随机选择权重并加载模型 ----------
    chosen_model_path = choose_random_model(weights_path, percents, pv_options, seed=seed)
    print(chosen_model_path)
    model = load_model_with_pv(chosen_model_path)

    # ---------- 设备 ----------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # ---------- 选车并评估（开环，基于库函数的逐帧样本） ----------
    ego_id = random.choice(tuple(scenario.id_list))
    res = test_predict_single_vehicle_openloop_with_pairs(model, scenario, ego_id, device=device, min_frames=13)

    if res is None:
        print("该车跳过")
    else:
        print("steps:", len(res["frames"]))
        print("MAE ax/ay:", res["mae"].tolist())
        print("RMSE ax/ay:", res["rmse"].tolist())

        # ===== 绘制 ax 真值 vs 预测值 =====
        ax_true = res["a_true"][:, 1].cpu().numpy()
        ax_pred = res["a_pred"][:, 1].cpu().numpy()
        frames = res["frames"].cpu().numpy()

        plt.figure(figsize=(10, 5))
        plt.plot(frames, ax_true, label="True ax", linewidth=1.5)
        plt.plot(frames, ax_pred, label="Predicted ax", linestyle="--", linewidth=1.5)
        plt.title(f"Ego {ego_id} - ax True vs Predicted (Open-loop, pairs)")
        plt.xlabel("Frame (index within this ego)")
        plt.ylabel("ax (m/s²)")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.show()
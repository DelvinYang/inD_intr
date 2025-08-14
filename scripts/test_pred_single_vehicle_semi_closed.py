import random
from pathlib import Path
from collections import deque
from typing import Dict, Optional, Tuple, List

import numpy as np
import torch
from loguru import logger
import matplotlib.pyplot as plt

from scripts.test import choose_random_model
from src.network import set_seed, load_model_with_pv
from src.read_data_inD import DataReaderInD
from frenet_system_creator import FrenetSystem
from src.scenarioind import ScenarioInD, Vehicle
from src.common import State, eight_dirs, dir_names


# ================= 工具：环境特征（真实） =================
@torch.no_grad()
def _build_env_dist_feature_real_cartesian(
    scenario: ScenarioInD,
    ego_frenet: FrenetSystem,
    ego_id: int | str,
    ego_state: State,
) -> Optional[torch.Tensor]:
    """环境 8 方向最近距离（真实自车 + 真实周车，Cartesian 距离）。"""
    ego_pos = np.array([ego_state.x[0], ego_state.y[0]])
    try:
        ds_ev, _ = ego_frenet.cartesian2ds_frame(ego_pos)
    except Exception as e:
        logger.warning(f"[ego {ego_id}] frenet transform failed: {e}")
        return None

    if not hasattr(ego_state, "frame_id"):
        raise RuntimeError("State must carry frame_id; set it in ScenarioInD.find_vehicle_state")

    dir_dist = {name: -1.0 for name in dir_names}
    svs_state = scenario.find_svs_state(ego_state.frame_id, ego_id)

    for sv in svs_state:
        sv_pos = np.array([sv.x[0], sv.y[0]])
        try:
            ds_sv, _ = ego_frenet.cartesian2ds_frame(sv_pos)
        except Exception:
            continue

        direction = eight_dirs(ds_ev, ds_sv, ego_state)
        if direction in dir_dist:
            dist = float(np.linalg.norm(sv_pos - ego_pos))
            if dir_dist[direction] == -1.0 or dist < dir_dist[direction]:
                dir_dist[direction] = dist

    return torch.tensor([dir_dist[name] for name in dir_names], dtype=torch.float32)  # [8]


@torch.no_grad()
def _build_feature_ego_env_real(
    scenario: ScenarioInD,
    ego_frenet: FrenetSystem,
    ego_id: int | str,
    ego_state: State,
) -> Optional[torch.Tensor]:
    """一帧完整特征 [12] = [vx, ax, vy, ay] + 8向距离（均为真实）。"""
    ego_feat = torch.tensor(
        [ego_state.lon[0], ego_state.lon[1], ego_state.lat[0], ego_state.lat[1]],
        dtype=torch.float32
    )
    env_feat = _build_env_dist_feature_real_cartesian(
        scenario, FrenetSystem(scenario.set_reference_path(ego_id)), ego_id, ego_state
    )
    if env_feat is None:
        return None
    return torch.cat([ego_feat, env_feat], dim=0)  # [12]


# ================ 半闭环（模式二，仅保留） ================
@torch.no_grad()
def test_predict_single_vehicle_semi_closed_acc_integrate(
    model: torch.nn.Module,
    scenario: ScenarioInD,
    ego_id: int | str,
    *,
    dt: float = 0.04,            # 帧间隔（如 25Hz -> 0.04）
    device: Optional[torch.device] = None,
    v_max: float = 25.0,         # 速度限幅（m/s）
    reset_every: int = 20,       # 每多少步纠偏一次
    reset_blend: float = 0.5,    # 纠偏权重：v = (1-b)*v + b*v_true
) -> Optional[Dict[str, torch.Tensor]]:
    """
    半闭环（模式二）：输入里的 ax, ay 用真值；vx, vy 用积分（v_t = v_{t-1} + a_true * dt），并周期性纠偏回真值速度。
    评价目标：预测“当前帧”的 [ax, ay] 与真值对比（从第13帧开始）。
    返回 None 表示帧不足或中途异常。
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------- 初始化 ----------
    try:
        ego_veh: Vehicle = scenario.find_vehicle_by_id(ego_id)
        ego_frenet = FrenetSystem(scenario.set_reference_path(ego_id))
    except Exception as e:
        logger.warning(f"[ego {ego_id}] init failed: {e}")
        return None

    start_f = ego_veh.initial_frame
    end_f   = ego_veh.final_frame
    T = end_f - start_f
    if T < 13:
        logger.info(f"[ego {ego_id}] frames={T} < 13, skip")
        return None

    model = model.to(device).eval()

    # ---------- 历史 12 帧（真实） ----------
    hist_seq: deque[torch.Tensor] = deque(maxlen=12)
    for t in range(start_f, start_f + 12):
        st = scenario.find_vehicle_state(t, ego_id)
        if st is None:
            logger.warning(f"[ego {ego_id}] missing frame {t}")
            return None
        if not hasattr(st, "frame_id"):
            st.frame_id = t
        feat_t = _build_feature_ego_env_real(scenario, ego_frenet, ego_id, st)
        if feat_t is None or not torch.isfinite(feat_t).all():
            logger.warning(f"[ego {ego_id}] bad feature at frame {t}")
            return None
        hist_seq.append(feat_t)

    # ---------- 准备积分速度 ----------
    st_11 = scenario.find_vehicle_state(start_f + 11, ego_id)
    if st_11 is None:
        return None
    v_lon = float(st_11.lon[0])
    v_lat = float(st_11.lat[0])

    # 预测记录
    pred_list: List[List[float]] = []
    true_list: List[List[float]] = []
    frames: List[int] = []

    # ---------- 逐帧预测 ----------
    for step, t in enumerate(range(start_f + 12, end_f), start=1):
        # 1) 输入 [1,12,12]
        X = torch.stack(list(hist_seq), dim=0).unsqueeze(0).to(device, dtype=torch.float32)

        # 2) 模型预测该帧的 [ax, ay]
        a_pred = model(X).squeeze(0).float().cpu()  # [2]
        ax_pred_raw, ay_pred_raw = float(a_pred[0]), float(a_pred[1])

        # 3) 真值
        st_t = scenario.find_vehicle_state(t, ego_id)
        if st_t is None:
            logger.warning(f"[ego {ego_id}] missing gt at frame {t}")
            return None
        if not hasattr(st_t, "frame_id"):
            st_t.frame_id = t
        ax_true, ay_true = float(st_t.lon[1]), float(st_t.lat[1])

        pred_list.append([ax_pred_raw, ay_pred_raw])
        true_list.append([ax_true, ay_true])
        frames.append(t)

        # 4) 生成下一步特征：速度积分 + 周期纠偏（环境始终真实）
        env_feat = _build_env_dist_feature_real_cartesian(scenario, ego_frenet, ego_id, st_t)
        if env_feat is None:
            return None

        # 用真 a 积分速度
        v_lon = np.clip(v_lon + ax_true * dt, -v_max, v_max)
        v_lat = np.clip(v_lat + ay_true * dt, -v_max, v_max)

        # 周期性纠偏回真速度（平滑过渡）
        if reset_every > 0 and (step % reset_every) == 0:
            v_lon = (1 - reset_blend) * v_lon + reset_blend * float(st_t.lon[0])
            v_lat = (1 - reset_blend) * v_lat + reset_blend * float(st_t.lat[0])

        ego_feat_next = torch.tensor(
            [v_lon, ax_true,   # 积分 vx + 真 ax
             v_lat, ay_true],  # 积分 vy + 真 ay
            dtype=torch.float32
        )

        feat_next = torch.cat([ego_feat_next, env_feat], dim=0)  # [12]
        hist_seq.append(feat_next)

    # ---------- 指标 ----------
    a_pred_t = torch.tensor(pred_list, dtype=torch.float32)  # [N,2]
    a_true_t = torch.tensor(true_list, dtype=torch.float32)  # [N,2]
    mae = torch.mean(torch.abs(a_pred_t - a_true_t), dim=0)
    mse = torch.mean((a_pred_t - a_true_t) ** 2, dim=0)
    rmse = torch.sqrt(mse)

    logger.info(f"[ego {ego_id} | semi-closed (acc-true, v-integrated)] steps={len(frames)} "
                f"MAE(ax,ay)=({mae[0]:.6f},{mae[1]:.6f}) "
                f"RMSE(ax,ay)=({rmse[0]:.6f},{rmse[1]:.6f})")

    return {
        "frames": torch.tensor(frames, dtype=torch.long),
        "a_pred": a_pred_t,
        "a_true": a_true_t,
        "mae": mae, "rmse": rmse, "mse": mse,
    }


# ================= 主入口 =================
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

    # ---------- 选车并评估（半闭环模式二） ----------
    ego_id = random.choice(tuple(scenario.id_list))
    res = test_predict_single_vehicle_semi_closed_acc_integrate(
        model, scenario, ego_id,
        dt=1.0 / scenario.frame_rate,
        device=device,
        v_max=25.0, reset_every=20, reset_blend=0.5
    )

    if res is None:
        print("该车跳过")
    else:
        print("steps:", len(res["frames"]))
        print("MAE ax/ay:", res["mae"].tolist())
        print("RMSE ax/ay:", res["rmse"].tolist())

        # ===== 绘制 ax 真值 vs 预测值 =====
        ax_true = res["a_true"][:, 0].cpu().numpy()
        ax_pred = res["a_pred"][:, 0].cpu().numpy()
        frames = res["frames"].cpu().numpy()

        plt.figure(figsize=(10, 5))
        plt.plot(frames, ax_true, label="True ax", linewidth=1.5)
        plt.plot(frames, ax_pred, label="Predicted ax", linestyle="--", linewidth=1.5)
        plt.title(f"Ego {ego_id} - ax True vs Predicted (Semi-closed: acc-true, v-integrated)")
        plt.xlabel("Frame")
        plt.ylabel("ax (m/s²)")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.show()
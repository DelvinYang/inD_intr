import random
from pathlib import Path

from matplotlib import pyplot as plt

from scripts.test import choose_random_model
from src.network import set_seed, load_model_with_pv
from src.read_data_inD import DataReaderInD


from collections import deque
from typing import Dict, Optional, Tuple
import numpy as np
import torch
from loguru import logger

from frenet_system_creator import FrenetSystem
from src.scenarioind import ScenarioInD, Vehicle
from src.common import State, eight_dirs, dir_names


@torch.no_grad()
def test_predict_single_vehicle_aggressive(
    model: torch.nn.Module,
    scenario: ScenarioInD,
    ego_id: int | str,
    dt: float = 0.04,                   # 采样周期（inD 常见 25Hz -> 0.04s）；如不同请调整
    device: Optional[torch.device] = None,
) -> Optional[Dict[str, torch.Tensor]]:
    """
    更激进的闭环测试（单车）：
      - 历史 12 帧用真实状态初始化；
      - 从第 13 帧起，预测 “下一帧(t)” 的 [ax, ay]；
      - 速度闭环：v_{t} = v_{t-1} + a_{t} * dt（lon/lat 两轴）；
      - 位置闭环（ds 坐标）：s_{t} = s_{t-1} + v_lon,t * dt， d_{t} = d_{t-1} + v_lat,t * dt；
      - 环境特征：用预测的自车 ds 位置 + 真实周车的 ds 位置，方向用 eight_dirs(ds_ev_pred, ds_sv, ego_state_t)，
        距离用 ds 空间的欧氏距离；
      - 特征帧写回序列时，自车 4 维用闭环的 [vx, ax, vy, ay]，八方向用基于预测位置计算的 8 维。
    返回 None 表示帧不足 13 或中途缺帧/异常。
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

    # ---------- 用真实状态构建历史 12 帧 ----------
    hist_seq = deque(maxlen=12)  # 每项 [12] 的 tensor
    for t in range(start_f, start_f + 12):
        state_t = scenario.find_vehicle_state(t, ego_id)
        if state_t is None:
            logger.warning(f"[ego {ego_id}] missing frame {t}")
            return None
        feat_t = _build_feature_ego_env_real(scenario, ego_frenet, ego_id, state_t)
        if feat_t is None:
            return None
        hist_seq.append(feat_t)  # 直接存 tensor，无后续拷贝

    # 当前闭环速度（以第 12 帧真实速度初始化）
    st_11 = scenario.find_vehicle_state(start_f + 11, ego_id)
    if st_11 is None:
        return None
    v_lon = float(st_11.lon[0])
    v_lat = float(st_11.lat[0])

    # 当前闭环位置（ds）：从第 12 帧的真实位置转换得到
    ego_pos_xy_11 = np.array([st_11.x[0], st_11.y[0]])
    ds_ev, _ = ego_frenet.cartesian2ds_frame(ego_pos_xy_11)
    s_pred, d_pred = float(ds_ev[0]), float(ds_ev[1])

    # ---------- 逐帧闭环预测 ----------
    pred_list, true_list, frames = [], [], []

    for t in range(start_f + 12, end_f):
        # (1) 组装输入 [1,12,12]，避免重复构造与告警
        X = torch.stack(list(hist_seq), dim=0).unsqueeze(0).to(device, dtype=torch.float32)

        # (2) 预测 t 帧的加速度
        a_pred = model(X).squeeze(0).float()          # [2]
        ax_pred, ay_pred = float(a_pred[0]), float(a_pred[1])

        # (3) 取真值用于评估
        st_t = scenario.find_vehicle_state(t, ego_id)
        if st_t is None:
            logger.warning(f"[ego {ego_id}] missing gt at frame {t}")
            return None
        ax_true, ay_true = float(st_t.lon[1]), float(st_t.lat[1])

        pred_list.append([ax_pred, ay_pred])
        true_list.append([ax_true, ay_true])
        frames.append(t)

        # (4) 闭环更新速度与 ds 位置（这一步决定“更激进”的环境重算）
        v_lon = v_lon + ax_pred * dt
        v_lat = v_lat + ay_pred * dt
        s_pred = s_pred + v_lon * dt
        d_pred = d_pred + v_lat * dt

        # (5) 用预测的 ds 位置 + 真实周车，重算环境 8 方向距离（ds 空间距离）
        env_feat = _build_env_dist_feature_pred_ds(
            scenario, ego_frenet, ego_id, st_t, ds_ev_pred=(s_pred, d_pred)
        )
        if env_feat is None:
            return None

        # (6) 写回下一帧要入队的特征（闭环 ego + 预测位置驱动的环境）
        ego_feat_closed = torch.tensor(
            [v_lon, ax_pred,  # lon: vx, ax
             v_lat, ay_pred], # lat: vy, ay
            dtype=torch.float32
        )
        feat_next = torch.cat([ego_feat_closed, env_feat], dim=0)  # [12]
        hist_seq.append(feat_next)

    # ---------- 评估指标 ----------
    a_pred_t = torch.tensor(pred_list, dtype=torch.float32)  # [N,2]
    a_true_t = torch.tensor(true_list, dtype=torch.float32)  # [N,2]
    mae = torch.mean(torch.abs(a_pred_t - a_true_t), dim=0)
    mse = torch.mean((a_pred_t - a_true_t) ** 2, dim=0)
    rmse = torch.sqrt(mse)

    logger.info(f"[ego {ego_id} | aggressive] steps={len(frames)} "
                f"MAE(ax,ay)=({mae[0]:.6f},{mae[1]:.6f}) "
                f"RMSE(ax,ay)=({rmse[0]:.6f},{rmse[1]:.6f})")

    return {
        "frames": torch.tensor(frames, dtype=torch.long),
        "a_pred": a_pred_t,
        "a_true": a_true_t,
        "mae": mae, "rmse": rmse, "mse": mse,
    }


def _build_feature_ego_env_real(
    scenario: ScenarioInD,
    ego_frenet: FrenetSystem,
    ego_id: int | str,
    ego_state: State,
) -> Optional[torch.Tensor]:
    """
    用真实状态构建一帧完整特征 [12]：
      ego(4) = [vx, ax, vy, ay]
      env(8) = 八方向最近距离（真实自车位置 + 真实周车，Cartesian 距离）
    仅用于初始化历史 12 帧。
    """
    ego_pos = np.array([ego_state.x[0], ego_state.y[0]])
    try:
        ds_ev, _ = ego_frenet.cartesian2ds_frame(ego_pos)
    except Exception as e:
        logger.warning(f"[ego {ego_id}] frenet transform failed: {e}")
        return None

    ego_feat = torch.tensor(
        [ego_state.lon[0], ego_state.lon[1], ego_state.lat[0], ego_state.lat[1]],
        dtype=torch.float32
    )
    # 环境：用真实位置与真实周车（Cartesian 距离），与原来一致
    env_feat = _build_env_dist_feature_real_cartesian(scenario, ego_frenet, ego_id, ego_state)
    if env_feat is None:
        return None

    return torch.cat([ego_feat, env_feat], dim=0)  # [12]


def _build_env_dist_feature_real_cartesian(
    scenario: ScenarioInD,
    ego_frenet: FrenetSystem,
    ego_id: int | str,
    ego_state: State,
) -> Optional[torch.Tensor]:
    """
    环境 8 方向距离（初始化阶段用）：真实自车 + 真实周车，采用 Cartesian 距离。
    """
    ego_pos = np.array([ego_state.x[0], ego_state.y[0]])
    try:
        ds_ev, _ = ego_frenet.cartesian2ds_frame(ego_pos)
    except Exception as e:
        logger.warning(f"[ego {ego_id}] frenet transform failed: {e}")
        return None

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
            dist = float(np.linalg.norm(sv_pos - ego_pos))  # Cartesian 距离（与历史一致）
            if dir_dist[direction] == -1.0 or dist < dir_dist[direction]:
                dir_dist[direction] = dist

    return torch.tensor([dir_dist[name] for name in dir_names], dtype=torch.float32)  # [8]


def _build_env_dist_feature_pred_ds(
    scenario: ScenarioInD,
    ego_frenet: FrenetSystem,
    ego_id: int | str,
    ego_state_t: State,
    ds_ev_pred: Tuple[float, float],
) -> Optional[torch.Tensor]:
    """
    环境 8 方向距离（预测阶段用）：预测自车 ds 位置 + 真实周车 ds 位置。
    - 方向：eight_dirs( ds_ev_pred, ds_sv, ego_state_t )  —— 朝向仍用真实 ego_state_t
    - 距离：ds 空间欧氏距离 sqrt(Δs^2 + Δd^2)
    """
    s_ev, d_ev = float(ds_ev_pred[0]), float(ds_ev_pred[1])

    dir_dist = {name: -1.0 for name in dir_names}
    svs_state = scenario.find_svs_state(ego_state_t.frame_id, ego_id)

    for sv in svs_state:
        sv_pos = np.array([sv.x[0], sv.y[0]])
        try:
            ds_sv, _ = ego_frenet.cartesian2ds_frame(sv_pos)
        except Exception:
            continue

        # 方向：用预测自车 ds 与真实周车 ds + 真实 ego_state 的朝向
        direction = eight_dirs((s_ev, d_ev), ds_sv, ego_state_t)
        if direction in dir_dist:
            # 距离：ds 空间
            ds_dist = float(np.hypot(ds_sv[0] - s_ev, ds_sv[1] - d_ev))
            if dir_dist[direction] == -1.0 or ds_dist < dir_dist[direction]:
                dir_dist[direction] = ds_dist

    return torch.tensor([dir_dist[name] for name in dir_names], dtype=torch.float32)  # [8]

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

    ego_id = random.choice(tuple(scenario.id_list))
    res = test_predict_single_vehicle_aggressive(model, scenario, ego_id, dt=1/scenario.frame_rate, device=device)
    if res is None:
        print("该车跳过")
    else:
        print("steps:", len(res["frames"]))
        print("MAE ax/ay:", res["mae"].tolist())
        print("RMSE ax/ay:", res["rmse"].tolist())

        # ===== 绘制 ax 真值 vs 预测值 =====
        ax_true = res["a_true"][:, 0].cpu().numpy()  # 真值 ax
        ax_pred = res["a_pred"][:, 0].cpu().numpy()  # 预测 ax
        frames = res["frames"].cpu().numpy()

        plt.figure(figsize=(10, 5))
        plt.plot(frames, ax_true, label="True ax", color="blue", linewidth=1.5)
        plt.plot(frames, ax_pred, label="Predicted ax", color="red", linestyle="--", linewidth=1.5)

        plt.title(f"Ego {ego_id} - ax True vs Predicted")
        plt.xlabel("Frame")
        plt.ylabel("ax (m/s²)")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.show()
import time
from typing import List

import numpy as np
import torch
from frenet_system_creator import FrenetSystem
from loguru import logger
from tqdm import tqdm

from src.common import State, eight_dirs, dir_names
from src.read_data_inD import DataReaderInD
from src.scenarioind import ScenarioInD, Vehicle

logger.info("start main")

prefix_number_list = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12',
                      '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24',
                      '25', '26', '27', '28', '29', '30', '31', '32']
data_path = '/Users/delvin/Desktop/programs/跨文化返修/inD'
past_frames_needed = 12

for prefix_number in prefix_number_list:
    try:
        data_reader = DataReaderInD(prefix_number, data_path)
    except:
        continue
    scenario = ScenarioInD(data_reader)

    all_sequence_samples = []


    for ego_id in tqdm(scenario.id_list, desc="Processing ego_ids"):
        ego_veh: Vehicle = scenario.find_vehicle_by_id(ego_id)
        try:
            ego_frenet_system = FrenetSystem(scenario.set_reference_path(ego_id))
        except:
            continue
        start_frame = ego_veh.initial_frame
        end_frame = ego_veh.final_frame

        total_available = end_frame - start_frame

        if total_available < past_frames_needed + 1:
            continue  # 不足 12+1 帧，跳过该车

        # === 顺序采样：采 12 帧，跳过 12 帧
        step = past_frames_needed + 12
        available_start_indices = list(range(start_frame, end_frame - past_frames_needed, step))

        # logger.info(f"ego_id {ego_id}: usable sequence start frames: {available_start_indices}")

        for idx in available_start_indices:
            frame_feature_list = []
            valid = True
            for offset in range(past_frames_needed):
                frame_num = idx + offset
                try:
                    ego_state: State = scenario.find_vehicle_state(frame_num, ego_id)
                    svs_state: List[State] = scenario.find_svs_state(frame_num, ego_id)

                    ego_pos = np.array([ego_state.x[0], ego_state.y[0]])
                    ds_pos_ev, _ = ego_frenet_system.cartesian2ds_frame(ego_pos)

                    # === ego 特征 ===
                    ego_feature = torch.tensor(
                        ego_state.lon + ego_state.lat,
                        dtype=torch.float
                    )  # [4]
                    dir_dist = {name: -1.0 for name in dir_names}
                    # logger.debug("ego_state: {}, svs_state: {}".format(ego_state, svs_state))
                    # === 周围车特征 ===
                    for sv in svs_state:
                        sv_pos = np.array([sv.x[0], sv.y[0]])
                        start = time.perf_counter()
                        ds_pos_sv, _ = ego_frenet_system.cartesian2ds_frame(sv_pos)
                        end = time.perf_counter()

                        # print(f"cartesian2ds_frame took {(end - start) * 1000:.3f} ms")
                        direction = eight_dirs(ds_pos_ev, ds_pos_sv, ego_state)
                        if direction in dir_dist:
                            dist = np.linalg.norm(sv_pos - ego_pos)
                            # 保留距离较近的（如果多个车在同一方向）
                            if dir_dist[direction] == -1.0 or dist < dir_dist[direction]:
                                dir_dist[direction] = dist

                    # 拼接最终特征
                    dist_tensor = torch.tensor([dir_dist[name] for name in dir_names], dtype=torch.float)
                    frame_feature = torch.cat([ego_feature, dist_tensor])  # [12]
                    frame_feature_list.append(frame_feature)

                except Exception as e:
                    logger.warning(f"skip sample at frame {frame_num} for ego {ego_id} due to {e}")
                    valid = False
                    break
            if valid:
                try:
                    target_state = scenario.find_vehicle_state(idx + past_frames_needed, ego_id)
                    label = torch.tensor([target_state.lon[1], target_state.lat[1]], dtype=torch.float)

                    sequence_tensor = torch.stack(frame_feature_list, dim=0)  # [T, 9, 4]
                    all_sequence_samples.append((sequence_tensor, label))
                except Exception as e:
                    logger.warning(f"skip label for ego {ego_id} due to {e}")
                    continue

    logger.info(f"Total samples collected: {len(all_sequence_samples)}")
    save_path = f"../data/sequence_tensor_dataset_{prefix_number}.pt"
    torch.save(all_sequence_samples, save_path)

    logger.info(f"Saved dataset with {len(all_sequence_samples)} samples to {save_path}")